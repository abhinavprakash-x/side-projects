#pragma once

#include <vector>
#include "Activation.h"
#include "Neuron.h"

using namespace std;

class Layer {
public:
    vector<Neuron> neurons;
    Activation activation;

    Layer(int neuronCount, int inputCount, Activation a);

    vector<double> calculate(const vector<double>& inputs);
};