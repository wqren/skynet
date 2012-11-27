/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <weights.cuh>
#include <mpi.h>
#include <boost/function.hpp>
#include <boost/bind.hpp>

#include "thread.h"
#include "logging.h"

static int64_t _bytesSent = 0;
static int64_t _bytesRecv = 0;
static double _timeWasted = 0;

bool Weights::_autoCopyToGPU = false;
WeightManager* WeightManager::_instance = NULL;

typedef map<int64_t, FreeList<Matrix> > MatrixFL;

class OutgoingWeights {
private:
    Matrix *_sending;
    Matrix *_pending;
    Matrix *_tmp;
    int64_t _id;
    
    vector<MPI::Request> _reqs;
public:
    OutgoingWeights(int64_t id, int numRows, int numCols) : _id(id) {
        _sending = new Matrix(numRows, numCols);
        _pending = new Matrix(numRows, numCols);
        _tmp = new Matrix(numRows, numCols);
    }

    void Add(const NVMatrix& m) {
      m.copyToHost(*_tmp);
      _pending->add(*_tmp);
    }

    void Send() {
        for (int i = 0; i < MPI::COMM_WORLD.Get_size(); ++i) {
            if (i == MPI::COMM_WORLD.Get_rank()) {
                continue;
            }
            // Log_Info("Sending batch... %d %d", _id, _out->getNumElements() * 4);
            _reqs.push_back(MPI::COMM_WORLD.Isend(_sending->getData(), _sending->getNumElements(), MPI::FLOAT, i, _id));
            _bytesSent += _sending->getNumElements() * 4;
        }
    }

    bool Finished() {
        return _reqs.empty() || MPI::Request::Testall(_reqs.size(), &_reqs[0]);
    }

    void swapPending() {
        _sending->scale(0);
        std::swap(_sending, _pending);
        _reqs.clear();
    }
};

class IncomingWeights {
private:
    MPI::Request _req;
    bool _started;
    int64_t _id;
    Matrix *_tgt;
public:
    IncomingWeights(int64_t id, Matrix* tgt) : _started(false), _id(id), _tgt(tgt) {
    }

    void StartRecv() {
      assert(!_started);
      _started = true;
      _req = MPI::COMM_WORLD.Irecv(_tgt->getData(), _tgt->getNumElements(), MPI::FLOAT, MPI::ANY_SOURCE, _id);
    }

    bool Finished() {
      MPI::Status stat;
      bool done = _req.Test(stat);
      if (!done) { return false; }
      Log_Assert(stat.Get_count(MPI::FLOAT) == _tgt->getNumElements(),
                 "Unexpected recv: %d %d %d", _id, _tgt->getNumElements() * 4, stat.Get_count(MPI::FLOAT));
      return true;
    }

    void Reset() {
      _started = false;
    }
};

static NVMatrix _gpuTmp = NULL;

struct WeightManager::WeightData {
    pthread_mutex_t sendMutex;
    pthread_mutex_t recvMutex;

    // NVMatrix inc;
    Matrix inc;
    bool incReady;

    Matrix recvTmp;
    OutgoingWeights* outgoing;
    IncomingWeights* incoming;

    int64_t id;

    WeightData(int64_t id, int numRows, int numCols) {
        pthread_mutex_init(&sendMutex, NULL);
        pthread_mutex_init(&recvMutex, NULL);
        recvTmp.resize(numRows, numCols);
        inc.resize(numRows, numCols);
        incoming = NULL;
        outgoing = new OutgoingWeights(id, numRows, numCols);

        this->id = id;
        incReady = false;
    }

    void handleRecv() {
        if (incoming == NULL) {
            incoming = new IncomingWeights(id, &recvTmp);
            incoming->StartRecv();
        }

        if (incoming->Finished()) {
            {
                // _gpuTmp is shared across WeightData instances, but is only used from the MPI thread.
                // _gpuTmp.resize(inc);
                // _gpuTmp.copyFromHost(recvTmp);
                
                ScopedLock l(recvMutex);
                // inc.add(_gpuTmp);
                inc.add(recvTmp);
                incReady = true;
            }
            incoming->Reset();
            incoming->StartRecv();
            _bytesRecv += recvTmp.getNumElements() * 4;
        }
    }

    void handleSend() {
        {
          ScopedLock l(sendMutex);
          if (outgoing->Finished()) {
            outgoing->swapPending();
            outgoing->Send();
          }
       }
    }
};

WeightManager* WeightManager::get() {
    if (_instance != NULL) {
        return _instance;
    }

    _instance = new WeightManager();
    return _instance;
}

WeightManager::WeightManager() {
    // _recvThread = new FuncThread(boost::bind(&WeightManager::_recvThreadFn, this));
    // _sendThread = new FuncThread(boost::bind(&WeightManager::_sendThreadFn, this));
}

void WeightManager::initialize() {
    WeightManager* w = WeightManager::get();
    assert(cudaGetDevice(&w->_cudaDevice) == cudaSuccess);
    w->_mpiThread = new FuncThread(boost::bind(&WeightManager::_mpiThreadFn, w));
}


void WeightManager::_mpiThreadFn() {
    Log_Info("Starting MPI worker thread, using CUDA device: %d", _cudaDevice);
    assert(cudaSetDevice(_cudaDevice) == cudaSuccess);
    cublasInit();
    while (1) {
        Sleep(0.01);
        for (int i = 0; i < _weights.size(); ++i) {
            WeightData* w = _weights[i];
            if (w == NULL) { 
                continue; 
            }

            w->handleRecv();
            w->handleSend();
        }
    }
}


void WeightManager::sendAndRecv(int64_t id, NVMatrix& delta, NVMatrix& weights) {
    weights.add(delta);

    TimerBlock tt(_timeWasted);
    if (!_weights[id]) {
        Log_Info("New weight vector %d - %d", id, delta.getNumElements() * 4);
        WeightData* w = new WeightData(id, delta.getNumRows(), delta.getNumCols());
        _weights[id] = w;
    }

    WeightData* w = _weights[id];
    {
        ScopedLock l(w->sendMutex);
        w->outgoing->Add(delta);
    }

    if (w->incReady) {
        ScopedLock l(w->recvMutex);
        assert(delta.getNumRows() == w->inc.getNumRows());
        assert(delta.getNumCols() == w->inc.getNumCols());

        _gpuTmp.resize(w->inc);
        _gpuTmp.copyFromHost(w->inc);
        weights.add(_gpuTmp);
        // weights.add(w->inc);
        w->inc.scale(0);
    }

    PERIODIC(5, Log_Info("MPI status: %.2fMB sent, %.2fMB received, %.2f seconds", _bytesSent / 1e6, _bytesRecv / 1e6, _timeWasted));
}

int64_t WeightManager::newId() {
    // Just insert an empty slot for now - the actually WeightData variables will be initialized
    // in sendUpdate, when we know the size of matrix to allocate.
    int64_t id = _weights.size();
    Log_Debug("New id: %d", id);
    _weights.push_back(NULL);
    return id;
}
