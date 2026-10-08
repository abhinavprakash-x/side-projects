#include "NeuralNetwork.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include "Loss.h"

using namespace std;

NeuralNetwork::NeuralNetwork(const vector<Layer>& nnlayers) : layers(nnlayers) {}

vector<double> NeuralNetwork::predict(const vector<double>& inputs)
{
    vector<double> current = inputs;

    for(auto& layer : layers)
        current = layer.calculate(current);

    return current;
}

double NeuralNetwork::calculate_loss(const vector<vector<double>>& inputs,
                                     const vector<vector<double>>& expected)
{
    if(inputs.size() != expected.size())
        throw invalid_argument("Input and expected dataset sizes differ");

    double totalLoss = 0.0;

    for(int i = 0; i < inputs.size(); ++i)
    {
        vector<double> output = predict(inputs[i]);
        totalLoss += mseLoss(output, expected[i]);
    }

    return totalLoss / inputs.size();
}

void NeuralNetwork::zero_gradients()
{
    for(auto& layer : layers)
    {
        for(auto& neuron : layer.neurons)
        {
            for(auto& gradient : neuron.weightGradients)
                gradient = 0.0;

            neuron.biasGradient = 0.0;
        }
    }
}

void NeuralNetwork::backward_propagation(const vector<double>& input,
                                         const vector<double>& expected)
{
    if(layers.empty())
        throw invalid_argument("Network has no layers");

    Layer& outputLayer = layers.back();

    if(outputLayer.neurons.size() != expected.size())
        throw invalid_argument("Expected output size does not match network output size");

    for(int i = 0; i < outputLayer.neurons.size(); ++i)
    {
        Neuron& neuron = outputLayer.neurons[i];
        double error = neuron.output - expected[i];

        neuron.delta = error * activationDerivative(neuron.z, outputLayer.activation);
    }

    for(int l = layers.size() - 2; l >= 0; --l)
    {
        Layer& currentLayer = layers[l];
        Layer& nextLayer = layers[l + 1];

        for(int j = 0; j < currentLayer.neurons.size(); ++j)
        {
            Neuron& neuron = currentLayer.neurons[j];
            double error = 0.0;

            for(int k = 0; k < nextLayer.neurons.size(); ++k)
                error += nextLayer.neurons[k].weights[j] * nextLayer.neurons[k].delta;

            neuron.delta = error * activationDerivative(neuron.z, currentLayer.activation);
        }
    }

    for(int l = 0; l < layers.size(); ++l)
    {
        Layer& layer = layers[l];
        vector<double> layerInput;

        if(l == 0)
            layerInput = input;
        else
        {
            for(auto& neuron : layers[l - 1].neurons)
                layerInput.push_back(neuron.output);
        }

        for(auto& neuron : layer.neurons)
        {
            for(int i = 0; i < neuron.weights.size(); ++i)
                neuron.weightGradients[i] += neuron.delta * layerInput[i];

            neuron.biasGradient += neuron.delta;
        }
    }
}

void NeuralNetwork::update_weights(double learningRate, int batchSize)
{
    for(auto& layer : layers)
    {
        for(auto& neuron : layer.neurons)
        {
            for(int i = 0; i < neuron.weights.size(); ++i)
                neuron.weights[i] -= learningRate * neuron.weightGradients[i] / batchSize;

            neuron.bias -= learningRate * neuron.biasGradient / batchSize;
        }
    }
}

void NeuralNetwork::train(const vector<vector<double>>& inputs,
                          const vector<vector<double>>& expected,
                          int epochs, double learningRate,
                          BatchType batchType,
                          int batchSize)
{
    if(inputs.empty())
        throw invalid_argument("Training dataset is empty");

    if(inputs.size() != expected.size())
        throw invalid_argument("Input and expected dataset sizes differ");

    if(batchType == BatchType::Stochastic)
        batchSize = 1;
    else if(batchType == BatchType::Batch)
        batchSize = inputs.size();
    else if(batchSize <= 0)
        throw invalid_argument("Mini-batch size must be greater than zero");

    batchSize = min(batchSize, (int)inputs.size());

    random_device rd;
    mt19937 gen(rd());

    vector<int> indices(inputs.size());
    iota(indices.begin(), indices.end(), 0);

    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        if(batchType == BatchType::Stochastic || batchType == BatchType::MiniBatch)
            shuffle(indices.begin(), indices.end(), gen);

        double totalLoss = 0.0;

        for(int start = 0; start < inputs.size(); start += batchSize)
        {
            int currentBatchSize = min(batchSize, (int)inputs.size() - start);

            zero_gradients();

            for(int i = start; i < start + currentBatchSize; ++i)
            {
                int index = indices[i];

                vector<double> output = predict(inputs[index]);
                totalLoss += mseLoss(output, expected[index]);

                backward_propagation(inputs[index], expected[index]);
            }

            update_weights(learningRate, currentBatchSize);
        }

        if(epoch % 1000 == 0)
            cout << "Epoch: " << epoch << " Loss: " << calculate_loss(inputs, expected) << endl;
    }
}