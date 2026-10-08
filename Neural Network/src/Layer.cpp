#include "Layer.h"

using namespace std;

Layer::Layer(int neuronCount, int inputCount, Activation a) : activation(a)
{
    for(int i = 0; i < neuronCount; ++i)
        neurons.emplace_back(inputCount);
}

vector<double> Layer::calculate(const vector<double>& inputs)
{
    vector<double> outputs;

    for(auto& neuron : neurons)
    {
        neuron.z = neuron.weightedSum(inputs);
        neuron.output = activate(neuron.z, activation);

        outputs.push_back(neuron.output);
    }

    return outputs;
}