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

#ifndef WEIGHTS_CUH
#define	WEIGHTS_CUH

#include <string>
#include <vector>
#include <iostream>
#include <cutil_inline.h>
#include <assert.h>
#include <nvmatrix.cuh>
#include <matrix.h>
#include <pthread.h>
#include "util.cuh"

using namespace std;

class FuncThread;
class OutgoingWeights;
class IncomingWeights;

struct WeightCombiner {
    double decay;
    double momentum;
    double learningRate;
    int numGradients;

    NVMatrix incTmp;

    WeightCombiner(double momentum, double decay, double learningRate);

    // Called when a new gradient is received (from the local or a remote machine).
    // The default behavior is to add gradients into the accumulator vector.
    virtual void newGradient(Matrix& gradient, Matrix& accumulator);
    virtual void newGradient(NVMatrix& gradient, NVMatrix& accumulator);

    // Optionally transform the outgoing gradient matrix before it is put on the network.
    virtual void transformGradient(NVMatrix& gradient) {
    }

    // Merge an accumulated set of gradients into our weight vector.
    virtual void apply(NVMatrix& weights, NVMatrix& grads, int numCases);
};

class AdagradCombiner: public WeightCombiner {
private:
    double _magnitude;

public:
    AdagradCombiner(double momentum, double decay, double learningRate) :
        WeightCombiner(momentum, decay, learningRate) {
        _magnitude = 1;
    }


    void newGradient(Matrix& gradient, Matrix& accumulator);
    void newGradient(NVMatrix& gradient, NVMatrix& accumulator);
    void apply(NVMatrix& weights, NVMatrix& grads, int numCases);
};

// Information about incoming/outgoing weight changes for a single layer.
struct WeightData {
    pthread_mutex_t sendMutex;
    pthread_mutex_t recvMutex;

    // NVMatrix inc;
    Matrix inc;
    bool incReady;
    int incCount;

    Matrix recvTmp;
    OutgoingWeights* outgoing;
    IncomingWeights* incoming;
    WeightCombiner* combiner;

    int64_t id;

    bool initialized;

    WeightData(int64_t id, WeightCombiner*);
    void initialize(int numRows, int numCols);
    bool handleRecv();
    bool handleSend();
};

typedef std::vector<WeightData*> WeightMap;

// Track incoming weight updates from remote machines.  Each Weight instance acquires a unique, consistent
// identifier via newId() at creation time.  sendUpdate should be called for each new weight delta created.
//
// Apply adds all current deltas to the given weight vector and resets the weight vector.
class NetworkManager {
private:
    static NetworkManager* _instance;
    NetworkManager();

    WeightMap _weights;

    int _cudaDevice;

    void _mpiThreadFn();
    FuncThread* _mpiThread;

    bool _pause;
    bool _isPaused;

    int64_t _bytesSent;
    int64_t _bytesRecv;
    double _timeWasted;

    NVMatrix _gpuTmp;
public:
    // Send out a new set of gradients, and apply any gradients received from remote machines
    // (as well as those passed in) to the weight matrix.
    void sendAndRecv(int64_t id, NVMatrix& gradients, NVMatrix& weights, int numCases);

    // Register this set of weights with the network manager.  The WeightCombiner determines
    // how to transform outgoing gradients, merge incoming gradients, and apply updates to
    // the weight vector.
    int64_t newId(WeightCombiner*);

    static NetworkManager* get();

    // Start the weight management threads.  Must be run from the GPU thread.
    static void initialize();

    static void pauseMPI();
    static void resumeMPI();
};

class Weights {
private:
    Matrix* _hWeights, *_hWeightsInc;
    NVMatrix* _weights, *_weightsInc, *_weightsGrad;

    float _epsW, _wc, _mom;
    bool _onGPU;
    int _numUpdates;
    static bool _autoCopyToGPU;
    int64_t _weightId;

    // Non-NULL if these weights are really shared from some other layer
    Weights* _srcWeights;

    NetworkManager *_netMgr;
    WeightCombiner *_weightCombiner;

public:
    NVMatrix& operator*() {
        return getW();
    }

    Weights(Weights& srcWeights, float epsW);
    Weights(Matrix& hWeights, Matrix& hWeightsInc, float epsW, float wc, float mom);
    ~Weights();

    static void setAutoCopyToGPU(bool autoCopyToGPU) {
        _autoCopyToGPU = autoCopyToGPU;
    }

    NVMatrix& getW() {
        assert(_onGPU);
        return *_weights;
    }

    NVMatrix& getGrad() {
        assert(_onGPU);
        return *_weightsGrad;
    }

    Matrix& getCPUW() {
        return *_hWeights;
    }

    Matrix& getCPUWInc() {
        return *_hWeightsInc;
    }

    int getNumRows() const {
        return _hWeights->getNumRows();
    }

    int getNumCols() const {
        return _hWeights->getNumCols();
    }

    void copyToCPU() {
        if (_srcWeights == NULL) {
            assert(_onGPU);
            _weights->copyToHost(*_hWeights);
            _weightsInc->copyToHost(*_hWeightsInc);
        }
    }

    // This function is assumed to be called in the order in which the layers
    // were defined
    void copyToGPU();

    void update(int numCases);

    int incNumUpdates() {
        if (_srcWeights != NULL) {
            return _srcWeights->incNumUpdates();
        }
        return _numUpdates++;
    }

    // Returns the number of times a gradient has been computed for this
    // weight matrix during the current pass (interval between two calls of update())
    // through the net. This number will only be greater than 1 if this weight matrix
    // is *shared* by multiple layers in the net.
    int getNumUpdates() const {
        if (_srcWeights != NULL) {
            return _srcWeights->getNumUpdates();
        }
        return _numUpdates;
    }

    float getEps() const {
        return _epsW;
    }

    float getMom() const {
        return _mom;
    }

    float getWC() const {
        return _wc;
    }
};

class WeightList {
private:
    std::vector<Weights*> _weightList;

public:
    Weights& operator[](const int idx) const {
        return *_weightList[idx];
    }

    ~WeightList() {
        for (int i = 0; i < _weightList.size(); i++) {
            delete _weightList[i];
        }
    }

    WeightList() {
    }

    void addWeights(Weights& w) {
        _weightList.push_back(&w);
    }

    void update(int numCases) {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->update(numCases);
        }
    }

    void copyToCPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToCPU();
        }
    }

    void copyToGPU() {
        for (int i = 0; i < getSize(); i++) {
            _weightList[i]->copyToGPU();
        }
    }

    int getSize() {
        return _weightList.size();
    }
};

#endif	/* WEIGHTS_CUH */
