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

bool Weights::_autoCopyToGPU = false;
WeightManager* WeightManager::_instance = NULL;

typedef map<int64_t, FreeList<Matrix> > MatrixFL;
static MatrixFL _sendTmp;

class SendBatch {
private:
    Matrix* _delta;
    int64_t _id;
    vector<MPI::Request> _reqs;
public:
    SendBatch(int64_t id, const NVMatrix& delta) :
                    _id(id) {
        _delta = _sendTmp[delta.getNumElements()].get();
        delta.copyToHost(*_delta);
        Log_Debug("Sending batch... %d", _id);
        for (int i = 0; i < MPI::COMM_WORLD.Get_size(); ++i) {
            if (i == MPI::COMM_WORLD.Get_rank()) {
                continue;
            }
            _reqs.push_back(MPI::COMM_WORLD.Isend(_delta->getData(), _delta->getNumElements(), MPI::FLOAT, i, _id));
        }
    }

    ~SendBatch() {
        _sendTmp[_delta->getNumElements()].release(_delta);
    }

    bool Finished() {
        return MPI::Request::Testall(_reqs.size(), &_reqs[0]);
    }

    void Wait() {
        MPI::Request::Waitall(_reqs.size(), &_reqs[0]);
        for (int i = 0; i < _reqs.size(); ++i) {
            _reqs[i].Free();
        }
    }
};

typedef vector<SendBatch*> OutList;

struct WeightManager::WeightData {
    pthread_mutex_t mutex;
    NVMatrix inc;
    NVMatrix tmp;
    OutList outgoing;
    MPI::Request incoming;

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
    _recvThread = new FuncThread(boost::bind(&WeightManager::_recvThreadFn, this));
    _sendThread = new FuncThread(boost::bind(&WeightManager::_sendThreadFn, this));
}

#define BEGIN_LOOP_OVER_WEIGHTS\
    for (int i = 0; i < _weights.size(); ++i) {\
                WeightData* w = _weights[i];\
                if (!w) { continue; }\
                ScopedLock l(w->mutex);

#define END_LOOP_OVER_WEIGHTS\
        Sleep(0.001);\
    }

void WeightManager::_recvThreadFn() {
    while (1) {
        BEGIN_LOOP_OVER_WEIGHTS
            MPI::Status stat;
            if (w->incoming.Test(stat)) {
                w->inc.add(w->tmp);
                w->incoming.Free();
                w->incoming = MPI::COMM_WORLD.Irecv(w->tmp.getDevData(), w->tmp.getNumElements(), MPI::FLOAT,
                                MPI::ANY_SOURCE, i);
            }

            END_LOOP_OVER_WEIGHTS
    }
}

void WeightManager::_sendThreadFn() {
    while (1) {
        BEGIN_LOOP_OVER_WEIGHTS
            for (OutList::iterator j = w->outgoing.begin(); j != w->outgoing.end();) {
                if ((*j)->Finished()) {
                    delete (*j);
                    j = w->outgoing.erase(j);
                } else {
                    ++j;
                }
            }
            END_LOOP_OVER_WEIGHTS
    }
}

void WeightManager::sendAndRecv(int64_t id, NVMatrix& delta, NVMatrix& weights) {
    Log_Info("Sending update: %d", id);
    if (!_weights[id]) {
        WeightData* w = new WeightData;
        w->tmp.resize(delta.getNumRows(), delta.getNumCols());
        w->inc.resize(delta.getNumRows(), delta.getNumCols());
        // Spin up our first receive for this data.
        w->incoming = MPI::COMM_WORLD.Irecv(w->tmp.getDevData(), w->tmp.getNumElements(), MPI::FLOAT, MPI::ANY_SOURCE,
                        id);
        _weights[id] = w;
    }

    SendBatch* b = new SendBatch(id, delta);

    WeightData* w = _weights[id];
    {
        ScopedLock l(w->mutex);
        w->outgoing.push_back(b);
        weights.add(w->inc);
        w->inc.scale(0);
    }

    weights.add(delta);
}

int64_t WeightManager::newId() {
    // Just insert an empty slot for now - the actually WeightData variables will be initialized
    // in sendUpdate, when we know the size of matrix to allocate.
    int64_t id = _weights.size();
    Log_Debug("New id: %d", id);
    _weights.push_back(NULL);
    return id;
}
