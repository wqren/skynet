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
// #include <boost/thread.hpp>
#include <mpi.h>

#include "common/logging.h"

bool Weights::_autoCopyToGPU = false;
MPI::Intracomm _world;

//boost::thread* _mpiThread;
WeightManager* WeightManager::_instance = NULL;

class MPIBatch {
private:
    Matrix* _data;
    int64_t _id;
    vector<MPI::Request> _reqs;

    bool _sent;
public:
    MPIBatch(int64_t id, Matrix *m) :
                    _id(id), _data(m), _sent(false) {
    }

    ~MPIBatch() {
        delete _data;
    }

    bool sent() {
        return _sent;
    }

    void Send() {
        Log_Debug("Sending batch... %d", _id);
        assert(!_sent);
        for (int i = 0; i < _world.Get_size(); ++i) {
            if (i == _world.Get_rank()) {
                continue;
            }
            _reqs.push_back(_world.Isend(_data->getData(), _data->getNumElements(), MPI::FLOAT, i, _id));
        }
        _sent = true;
    }

    bool Finished() {
        return MPI::Request::Testall(_reqs.size(), &_reqs[0]);
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
    //MPI::Init_thread(MPI::THREAD_MULTIPLE);
    _world = MPI::COMM_WORLD;

//    _mpiThread = new boost::thread(&WeightManager::run, this);
}

void WeightManager::sendUpdate(int64_t id, NVMatrix& m) {
    Log_Info("Sending update: %d", id);
    Matrix *hostM = new Matrix;
    hostM->resize(m.getNumRows(), m.getNumCols());
    m.copyToHost(*hostM);

    if (!_inc[id]) {
        _inc[id] = new Matrix(m.getNumRows(), m.getNumCols());
    }
    _inc[id]->add(*hostM);

    MPIBatch *b = new MPIBatch(id, hostM);
    b->Send();
    _batches.push_back(b);
}

void WeightManager::fetchUpdates(int64_t id) {
    Log_Debug("Fetching updates... %d", id);

    MPI::Status stat;
    while (_world.Iprobe(MPI::ANY_SOURCE, id, stat)) {
        int64_t source = stat.Get_source();

        Log_Info("MPI:: recv from %d.%d -- %.5f MB", source, id, stat.Get_count(MPI::BYTE) / 1e6);

        Matrix tmp(1, stat.Get_count(MPI::FLOAT));
        _world.Recv(tmp.getData(), stat.Get_count(MPI::FLOAT), MPI::FLOAT, source, id);

        Matrix* tgt = _inc[id];
        assert(tgt != NULL), "Missing target matrix?!";
        tmp.reshape(tgt->getNumRows(), tgt->getNumCols());
        tgt->add(tmp);
    }

    for (list<MPIBatch*>::iterator i = _batches.begin(); i != _batches.end();) {
        MPIBatch* b = *i;
        if (b->Finished()) {
            delete b;
            i = _batches.erase(i);
        } else {
            ++i;
        }
    }
}

void WeightManager::applyUpdates(int64_t id, NVMatrix& weights) {
    fetchUpdates(id);

    Log_Debug("Applying updates: %d", id);
    NVMatrix devM;
    devM.resize(weights);

    devM.copyFromHost(*_inc[id]);
    weights.add(devM);
    _inc[id]->scale(0);
}

int64_t WeightManager::newId() {
    int64_t id = _inc.size();
    Log_Debug("New id: %d", id);
    _inc[id] = NULL;
    return id;
}
