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

#include "common/logging.h"

bool Weights::_autoCopyToGPU = false;
MPI::Intracomm _world;

WeightManager* WeightManager::_instance = NULL;

static double _bytesSent = 0;
static double _bytesRecv = 0;
static double _timeWasted = 0;

class SendBatch {
private:
    NVMatrix* _m;
    int64_t _id;
    vector<MPI::Request> _reqs;
public:
    SendBatch(int64_t id, NVMatrix *m) :
                    _id(id), _m(m) {
    }

    void Send() {
        TimerBlock tt(_timeWasted);
        Log_Debug("Sending batch... %d", _id);
        for (int i = 0; i < _world.Get_size(); ++i) {
            if (i == _world.Get_rank()) {
                continue;
            }
            // Log_Info("%d sending to %d", _world.Get_rank(), i);
            _reqs.push_back(_world.Isend(_m->getDevData(), _m->getNumElements(), MPI::FLOAT, i, _id));
            _bytesSent += _m->getNumElements() * 4;
        }
    }

    bool Finished() {
        return MPI::Request::Testall(_reqs.size(), &_reqs[0]);
    }

    void Wait() {
        TimerBlock tt(_timeWasted);
        MPI::Request::Waitall(_reqs.size(), &_reqs[0]);
        for (int i = 0; i < _reqs.size(); ++i) {
            _reqs[i].Free();
        }
    }
};

typedef map<int64_t, SendBatch*> SendMap;
struct WeightManager::Context {
    NVMatrix inc;
    NVMatrix tmp;
    SendMap outgoing;
    MPI::Request incoming;
};

WeightManager* WeightManager::get() {
    if (_instance != NULL) {
        return _instance;
    }

    _instance = new WeightManager();
    return _instance;
}

WeightManager::WeightManager() {
    _world = MPI::COMM_WORLD;
}

void WeightManager::sendUpdate(int64_t id, NVMatrix& m) {
    Log_Debug("Sending update: %d", id);
    Context* ctx = _ctx[id];
    if (!ctx) {
        ctx = new Context;
        _ctx[id] = ctx;
        ctx->tmp.resize(m.getNumRows(), m.getNumCols());
        ctx->inc.resize(m.getNumRows(), m.getNumCols());

        ctx->inc.scale(0);
        ctx->tmp.scale(0);

        // Spin up our first receive for this data.
        ctx->incoming = _world.Irecv(ctx->tmp.getDevData(), ctx->tmp.getNumElements(), MPI::FLOAT, MPI::ANY_SOURCE, id);
    }

    ctx->inc.add(m);

    if (ctx->outgoing[id] != NULL) {
        Log_Debug("Waiting for %d", id);
        ctx->outgoing[id]->Wait();
    }

    SendBatch *b = new SendBatch(id, &m);
    b->Send();
    ctx->outgoing[id] = b;
}

void WeightManager::applyUpdates(int64_t id, NVMatrix& weights) {
    Log_Debug("Fetching updates... %d", id);
    Context *ctx = _ctx[id];
    // We finished receiving data - increment our local weight delta and open up for another receive.
    MPI::Status stat;
    if (ctx->incoming.Test(stat)) {
        //Log_Info("Error: %d", stat.Get_error());
       // Log_Info("Adding received data to increment: %d, %d", stat.Get_count(MPI::FLOAT), stat.Get_elements(MPI::FLOAT));
        TimerBlock tt(_timeWasted);
        ctx->inc.add(ctx->tmp);
        assert(stat.Get_count(MPI::FLOAT) == ctx->tmp.getNumElements());
        
        ctx->incoming.Free();
        ctx->incoming = _world.Irecv(ctx->tmp.getDevData(), ctx->tmp.getNumElements(), MPI::FLOAT, MPI::ANY_SOURCE, id);
        _bytesRecv += ctx->tmp.getNumElements() * 4;
    }

    weights.add(ctx->inc);
    ctx->inc.scale(0);

    PERIODIC(1, Log_Info("MPI status: %.2f MB sent, %.2f MB recv, wasted time: %.2f sec", _bytesSent / 1e6, _bytesRecv / 1e6, _timeWasted));
}

int64_t WeightManager::newId() {
    // Just insert an empty slot for now - the actually context variables will be initialized
    // in sendUpdate, when we know the size of matrix to allocate.
    int64_t id = _ctx.size();
    Log_Debug("New id: %d", id);
    _ctx[id] = NULL;
    return id;
}
