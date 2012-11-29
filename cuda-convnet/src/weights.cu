/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * Copyright (c) 2012, Russell Power (russell.power@gmail.com)
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
#include <pthread.h>

#include <vector>
#include <map>

#include "thread.h"
#include "logging.h"

bool Weights::_autoCopyToGPU = false;

void Weights::copyToGPU() {
    if (_srcWeights == NULL) {
        _weights = new NVMatrix();
        _weightsInc = new NVMatrix();
        _weightsGrad = new NVMatrix();
        _weights->copyFromHost(*_hWeights, true);
        _weightsInc->copyFromHost(*_hWeightsInc, true);
//        _weightsGrad->resize(_weightsInc->getNumRows(), _weightsInc->getNumCols());
        _weightsGrad->resize(*_weightsInc);
        Log_Info("Gradients resized to %d %d", _weightsGrad->getNumRows(), _weightsGrad->getNumCols());
        Log_Info("Weights are sized: %d %d", _weightsInc->getNumRows(), _weightsInc->getNumCols());
    } else {
        _weights = _srcWeights->_weights;
        _weightsInc = _srcWeights->_weightsInc;
        _weightsGrad = _srcWeights->_weightsGrad;
    }
    _onGPU = true;
}

NetworkManager* NetworkManager::_instance = NULL;

using namespace std;

typedef map<int64_t, FreeList<Matrix> > MatrixFL;

// We use 2 weight vectors to send out data, 'sending' and 'pending'.
// The network thread attempts to push data from 'sending' as fast
// as possible, whether or not updates have been created.
//
// Whenever a new delta is produced, we update the 'pending' vector.
//
// As soon as a batch of updates is sent, the 'sending' and 'pending'
// vectors are swapped.
class OutgoingWeights {
private:
    Matrix *_sending;
    Matrix *_pending;
    Matrix *_tmp;
    int64_t _id;

    vector<MPI::Request> _reqs;
public:
    OutgoingWeights(int64_t id, int numRows, int numCols) :
                    _id(id) {
        _sending = new Matrix(numRows, numCols);
        _pending = new Matrix(numRows, numCols);
        _tmp = new Matrix(numRows, numCols);
    }

    void addDelta(const NVMatrix& m) {
        m.copyToHost(*_tmp);
        _pending->add(*_tmp);
    }

    void startSend() {
        for (int i = 0; i < MPI::COMM_WORLD.Get_size(); ++i) {
            if (i == MPI::COMM_WORLD.Get_rank()) {
                continue;
            }
            // Log_Info("Sending batch... %d %d", _id, _out->getNumElements() * 4);
            _reqs.push_back(MPI::COMM_WORLD.Isend(_sending->getData(), _sending->getNumElements(), MPI::FLOAT, i, _id));
        }
    }

    bool sendDone() {
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
    IncomingWeights(int64_t id, Matrix* tgt) :
                    _started(false), _id(id), _tgt(tgt) {
    }

    void startRecv() {
        assert(!_started);
        _started = true;
        _req = MPI::COMM_WORLD.Irecv(_tgt->getData(), _tgt->getNumElements(), MPI::FLOAT, MPI::ANY_SOURCE, _id);
    }

    bool recvDone() {
        MPI::Status stat;
        bool done = _req.Test(stat);
        if (!done) {
            return false;
        }
        Log_Assert(stat.Get_count(MPI::FLOAT) == _tgt->getNumElements(), "Unexpected recv: %d %d %d",
                        _id, _tgt->getNumElements() * 4, stat.Get_count(MPI::FLOAT));
        return true;
    }

    void reset() {
        _started = false;
    }
};

WeightData::WeightData(int64_t id, int numRows, int numCols) {
    pthread_mutex_init(&sendMutex, NULL);
    pthread_mutex_init(&recvMutex, NULL);
    outgoing = new OutgoingWeights(id, numRows, numCols);
    this->id = id;

    recvTmp.resize(numRows, numCols);
    inc.resize(numRows, numCols);
    incReady = false;
    incoming = NULL;
    incCount = 0;
}

bool WeightData::handleRecv() {
    if (incoming == NULL) {
        incoming = new IncomingWeights(id, &recvTmp);
        incoming->startRecv();
    }
    if (incoming->recvDone()) {
        {
            // _gpuTmp is shared across WeightData instances, but is only used from the MPI thread.
            // _gpuTmp.resize(inc);
            // _gpuTmp.copyFromHost(recvTmp);
            ScopedLock l(recvMutex);
            // inc.add(_gpuTmp);
            inc.add(recvTmp);
            incReady = true;
        }
        incoming->reset();
        incoming->startRecv();
        return true;
    }
    return false;
}

bool WeightData::handleSend() {
    {
        ScopedLock l(sendMutex);
        if (outgoing->sendDone()) {
            outgoing->swapPending();
            outgoing->startSend();
            return true;
        }
        return false;
    }
}

NetworkManager* NetworkManager::get() {
    if (_instance != NULL) {
        return _instance;
    }

    _instance = new NetworkManager();
    return _instance;
}

NetworkManager::NetworkManager() {
    _cudaDevice = -1;
    _pause = _isPaused = false;
    _mpiThread = NULL;
    _bytesRecv = 0;
    _bytesSent = 0;
    _timeWasted = 0;
}

void NetworkManager::initialize() {
    NetworkManager* w = NetworkManager::get();
    assert(cudaGetDevice(&w->_cudaDevice) == cudaSuccess);
    w->_mpiThread = new FuncThread(boost::bind(&NetworkManager::_mpiThreadFn, w));
}

void NetworkManager::pauseMPI() {
    NetworkManager::get()->_pause = true;
    while (!NetworkManager::get()->_isPaused) {
        Sleep(0.001);
    }

    Log_Debug("MPI thread paused.");
}

void NetworkManager::resumeMPI() {
    NetworkManager::get()->_pause = false;
    while (NetworkManager::get()->_isPaused) {
        Sleep(0.001);
    }

    Log_Debug("MPI thread resumed.");
}

void NetworkManager::_mpiThreadFn() {
    Log_Info("Starting MPI worker thread, using CUDA device: %d", _cudaDevice);
    assert(cudaSetDevice(_cudaDevice) == cudaSuccess);
    cublasInit();
    while (1) {
        Sleep(0.01);
        if (_pause) {
            _isPaused = true;
            continue;
        }

        _isPaused = false;
        for (int i = 0; i < _weights.size(); ++i) {
            WeightData* w = _weights[i];
            if (w == NULL) {
                continue;
            }

            if (w->handleRecv()) {
                _bytesRecv += w->recvTmp.getNumDataBytes();
            }
            if (w->handleSend()) {
                _bytesSent += w->recvTmp.getNumDataBytes() * (MPI::COMM_WORLD.Get_size() - 1);
            }
        }
    }
}

void NetworkManager::sendAndRecv(int64_t id, NVMatrix& delta, NVMatrix& weights) {
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
        w->outgoing->addDelta(delta);
    }

    if (w->incReady) {
        ScopedLock l(w->recvMutex);
        assert(delta.getNumRows() == w->inc.getNumRows());
        assert(delta.getNumCols() == w->inc.getNumCols());

        _gpuTmp.resize(w->inc);
        _gpuTmp.copyFromHost(w->inc);
        _gpuTmp.add(delta);
        weights.add(_gpuTmp, 1 / (1.0 + w->incCount));

        // weights.add(_gpuTmp);

        // w->inc.add(delta);
        // weights.add(w->inc);
        w->inc.scale(0);
        w->incCount = 0;
    } else {
        weights.add(delta);
    }

    PERIODIC(5,
                    Log_Info("MPI status: %.2fMB sent, %.2fMB received, %.2f seconds", _bytesSent / 1e6, _bytesRecv / 1e6, _timeWasted));
}

int64_t NetworkManager::newId() {
    // Just insert an empty slot for now - the actually WeightData variables will be initialized
    // in sendUpdate, when we know the size of matrix to allocate.
    int64_t id = _weights.size();
    Log_Debug("New id: %d", id);
    _weights.push_back(NULL);
    return id;
}
