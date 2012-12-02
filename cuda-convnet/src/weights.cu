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
#include <math.h>

#include <vector>
#include <map>

#include "thread.h"
#include "logging.h"

bool Weights::_autoCopyToGPU = false;

WeightCombiner::WeightCombiner(double momentum, double decay, double learningRate) :
                momentum(momentum), decay(decay), learningRate(learningRate), numGradients(0), magnitude(0) {
}

void WeightCombiner::newGradient(Matrix& gradient, Matrix& accumulator) {
    if (numGradients == 0) {
      gradient.copy(accumulator);
    } else {
      accumulator.add(gradient);
    }
    magnitude += gradient.norm2();
    ++numGradients;
}

void WeightCombiner::newGradient(NVMatrix& gradient, NVMatrix& accumulator) {
    if (numGradients == 0) {
      gradient.copy(accumulator);
    } else {
      accumulator.add(gradient);
    }
    magnitude += gradient.norm2();
    ++numGradients;
}

void WeightCombiner::apply(NVMatrix& weights, NVMatrix& previous, NVMatrix& grads, int numCases) {
    incTmp.resize(weights);
    grads.scale(learningRate / numCases, incTmp);

    if (momentum > 0) {
        Log_Debug("%.10f - momentum", momentum);
        incTmp.add(previous, momentum);
    }

    if (decay > 0) {
        Log_Debug("%.10f - decay", momentum);
        incTmp.add(weights, -decay * learningRate);
    }
   
    // Log_Info("%f %f %f %f %f %f", learningRate, numCases, momentum, decay, previous.norm2(), grads.norm2());
    weights.add(incTmp);
    incTmp.copy(previous);

    // assert(!isnan(incTmp.norm2()));
    // assert(!isnan(weights.norm2()));
    // assert(weights.norm2() > 0);
    numGradients = 0;
}

Weights::Weights(Weights& srcWeights, float epsW) :
                _srcWeights(&srcWeights), _epsW(epsW), _wc(0), _onGPU(false), _numUpdates(0), _weights(NULL), _weightsInc(
                                NULL), _weightsGrad(NULL) {
    _hWeights = &srcWeights.getCPUW();
    _hWeightsInc = &srcWeights.getCPUWInc();
    _mom = srcWeights.getMom();
    _netMgr = NetworkManager::get();
    _weightId = _netMgr->newId(new WeightCombiner(_mom, _wc, _epsW));
    if (_autoCopyToGPU) {
        copyToGPU();
    }
}

Weights::Weights(Matrix& hWeights, Matrix& hWeightsInc, float epsW, float wc, float mom) :
                _srcWeights(NULL), _hWeights(&hWeights), _hWeightsInc(&hWeightsInc), _numUpdates(0), _epsW(epsW), _wc(
                                wc), _mom(mom), _onGPU(false), _weights(NULL), _weightsInc(NULL), _weightsGrad(NULL) {
    _netMgr = NetworkManager::get();
    _weightId = _netMgr->newId(new WeightCombiner(_mom, _wc, _epsW));
    if (_autoCopyToGPU) {
        copyToGPU();
    }
}

Weights::~Weights() {
    delete _hWeights;
    delete _hWeightsInc;
    if (_srcWeights == NULL) {
        delete _weights;
        delete _weightsInc;
        delete _weightsGrad;
    }
}

void Weights::copyToGPU() {
    if (_srcWeights == NULL) {
        if (_weights == NULL) {
            _weights = new NVMatrix();
            _weightsInc = new NVMatrix();
            _weightsGrad = new NVMatrix();
        }

        _weights->copyFromHost(*_hWeights, true);
        _weightsInc->copyFromHost(*_hWeightsInc, true);
        _weightsGrad->zero();
    } else {
        _weights = _srcWeights->_weights;
        _weightsInc = _srcWeights->_weightsInc;
        _weightsGrad = _srcWeights->_weightsGrad;
    }
    _onGPU = true;

    Log_Debug("%f %f", _weights->norm2(), _weightsInc->norm2());
}

void Weights::update(int numCases) {
    // Only true owner of weights updates
    if (_srcWeights == NULL && _epsW > 0) {
        assert(_onGPU);

        _netMgr->sendAndRecv(_weightId, *_weightsGrad, *_weightsInc, *_weights, numCases);
        _weightsGrad->scale(0);
        // assert(!isnan(_weightsGrad->norm2()));
        _numUpdates = 0;
    } else {
      // Log_Info("Skipping update...");
    }
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
            // Log_Debug("Sending batch... %d %d", _id, _out->getNumElements() * 4);
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

WeightData::WeightData(int64_t id, WeightCombiner* combiner) {
    pthread_mutex_init(&sendMutex, NULL);
    pthread_mutex_init(&recvMutex, NULL);

    incReady = false;
    incCount = 0;
    this->id = id;

    incoming = NULL;
    outgoing = NULL;
    this->combiner = combiner;

    initialized = false;
}

void WeightData::initialize(int numRows, int numCols) {
    outgoing = new OutgoingWeights(id, numRows, numCols);
    recvTmp.resize(numRows, numCols);
    inc.resize(numRows, numCols);
    inc.scale(0);
    initialized = true;
}

bool WeightData::handleRecv() {
    if (incoming == NULL) {
        incoming = new IncomingWeights(id, &recvTmp);
        incoming->startRecv();
    }
    if (incoming->recvDone()) {
        {
            ScopedLock l(recvMutex);
            combiner->newGradient(recvTmp, inc);
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
    Log_Debug("Starting MPI worker thread, using CUDA device: %d", _cudaDevice);
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
            if (!w->initialized) {
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

void NetworkManager::sendAndRecv(int64_t id, NVMatrix& gradient, NVMatrix& increment, NVMatrix& weights, int numCases) {
    TimerBlock tt(_timeWasted);

    WeightData* w = _weights[id];
    w->combiner->transformGradient(gradient);
    
    if (!w->initialized) {
        ScopedLock lw(w->sendMutex);
        ScopedLock lr(w->recvMutex);
        w->initialize(gradient.getNumRows(), gradient.getNumCols());
    }

    if (MPI::COMM_WORLD.Get_size() > 1) {
        ScopedLock l(w->sendMutex);
        w->outgoing->addDelta(gradient);
    }

    if (w->incReady) {
        ScopedLock l(w->recvMutex);
        assert(gradient.getNumRows() == w->inc.getNumRows());
        assert(gradient.getNumCols() == w->inc.getNumCols());

        _gpuTmp.resize(w->inc);
        _gpuTmp.copyFromHost(w->inc);
        w->combiner->newGradient(gradient, _gpuTmp);
        w->combiner->apply(weights, increment, _gpuTmp, numCases);
        w->inc.scale(0);
        w->incCount = 0;
    } else {
        _gpuTmp.resize(gradient);
        w->combiner->newGradient(gradient, _gpuTmp);
        w->combiner->apply(weights, increment, _gpuTmp, numCases);
    }

    // PERIODIC(30, Log_Debug("MPI status: %.2fMB sent, %.2fMB received, %.2f seconds", _bytesSent / 1e6, _bytesRecv / 1e6, _timeWasted));
}

int64_t NetworkManager::newId(WeightCombiner* combiner) {
    int64_t id = _weights.size();
    _weights.push_back(new WeightData(id, combiner));
    return id;
}
