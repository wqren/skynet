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
static MatrixFL _sendTmp;

class OutgoingWeights {
private:
    Matrix* _delta;
    int64_t _id;
    bool _sent;
    vector<MPI::Request> _reqs;
public:
    OutgoingWeights(int64_t id, const NVMatrix& delta) :
                    _id(id), _sent(false) {
        TimerBlock tt(_timeWasted);
        _delta = _sendTmp[delta.getNumElements()].get();
        _delta->resize(delta.getNumRows(), delta.getNumCols());
        delta.copyToHost(*_delta);
    }

    ~OutgoingWeights() {
        _sendTmp[_delta->getNumElements()].release(_delta);
    }

    bool getSent() { return _sent; }

    void Send() {
        assert(!_sent);
        _sent = true;
        for (int i = 0; i < MPI::COMM_WORLD.Get_size(); ++i) {
            if (i == MPI::COMM_WORLD.Get_rank()) {
                continue;
            }
            // Log_Info("Sending batch... %d %d", _id, _delta->getNumElements() * 4);
            _reqs.push_back(MPI::COMM_WORLD.Isend(_delta->getData(), _delta->getNumElements(), MPI::FLOAT, i, _id));
            _bytesSent += _delta->getNumElements() * 4;
        }
    }

    bool Finished() {
        return MPI::Request::Testall(_reqs.size(), &_reqs[0]);
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

typedef vector<OutgoingWeights*> OutList;

struct WeightManager::WeightData {
    pthread_mutex_t mutex;
    Matrix inc;
    Matrix tmp;
    OutList outgoing;
    IncomingWeights* incoming;

    WeightData() {
        pthread_mutex_init(&mutex, NULL);
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
    _mpiThread = new FuncThread(boost::bind(&WeightManager::_mpiThreadFn, this));
}


void WeightManager::_mpiThreadFn() {
    while (1) {
        Sleep(0.01);
        for (int i = 0; i < _weights.size(); ++i) {
            WeightData* w = _weights[i];
            if (w == NULL) { 
                continue; 
            }

            // check for and receive incoming data...
            ScopedLock l(w->mutex);
            if (w->incoming == NULL) {
                w->incoming = new IncomingWeights(i, &w->tmp);
                w->incoming->StartRecv();
            }

            if (w->incoming->Finished()) {
                w->inc.add(w->tmp);
                _bytesRecv += w->tmp.getNumElements() * 4;
                w->incoming->Reset();
                w->incoming->StartRecv();
            }

            for (OutList::iterator j = w->outgoing.begin(); j != w->outgoing.end();) {
                OutgoingWeights* o = *j;
                if (!o->getSent()) {
                    o->Send();
                }

                if (o->Finished()) {
                    delete o;
                    j = w->outgoing.erase(j);
                } else {
                    ++j;
                }
            }
        }
    }
}


void WeightManager::sendAndRecv(int64_t id, NVMatrix& delta, NVMatrix& weights) {
    weights.add(delta);

    TimerBlock tt(_timeWasted);
    if (!_weights[id]) {
        Log_Info("New weight vector %d - %d", id, delta.getNumElements() * 4);
        WeightData* w = new WeightData;
        w->tmp.resize(delta.getNumRows(), delta.getNumCols());
        w->inc.resize(delta.getNumRows(), delta.getNumCols());
        w->incoming = NULL;
        _weights[id] = w;
    }

    OutgoingWeights* b = new OutgoingWeights(id, delta);
    WeightData* w = _weights[id];
    {
        ScopedLock l(w->mutex);
        assert(delta.getNumRows() == w->tmp.getNumRows());
        assert(delta.getNumCols() == w->tmp.getNumCols());

        w->outgoing.push_back(b);

        _addTmp.resize(w->inc);
        _addTmp.copyFromHost(w->inc);
        weights.add(_addTmp);
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
