#include "Neuron.h"

#include <cstdlib>
#include <stdexcept>

using namespace std;

Neuron::Neuron(int inputCount)
{
    weights.resize(inputCount);
    weightGradients.resize(inputCount);

    for(int i = 0; i < inputCount; ++i)
    {
        weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        weightGradients[i] = 0.0;
    }

    bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    z = 0.0;
    output = 0.0;
    delta = 0.0;
    biasGradient = 0.0;
}

double Neuron::weightedSum(const vector<double>& x)
{
    if(x.size() != weights.size())
        throw invalid_argument("Input size does not match neuron weights");

    z = bias;

    for(int i = 0; i < weights.size(); ++i)
        z += weights[i] * x[i];

    return z;
}