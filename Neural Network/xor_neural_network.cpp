#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <algorithm>

using namespace std;

enum class Activation {
    ReLU,
    Sigmoid,
    Tanh
};

double ReLU(double z)
{
    return max(0.0, z);
}

double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

double Tanh(double z)
{
    return tanh(z);
}

double activate(double z, Activation type)
{
    switch(type)
    {
        case Activation::ReLU: return ReLU(z);
        case Activation::Sigmoid: return sigmoid(z);
        case Activation::Tanh: return Tanh(z);
        default: throw invalid_argument("Invalid Activation function");
    }
}

double activationDerivative(double z, Activation type)
{
    switch(type)
    {
        case Activation::Sigmoid:
        {
            double s = sigmoid(z);
            return s * (1.0 - s);
        }
        case Activation::Tanh:
        {
            double t = Tanh(z);
            return 1.0 - t * t;
        }
        case Activation::ReLU: return z > 0.0 ? 1.0 : 0.0;
        default: throw invalid_argument("Invalid Activation function");
    }
}

double mseLoss(const vector<double>& predicted, const vector<double>& expected)
{
    if(predicted.size() != expected.size())
        throw invalid_argument("Prediction and expected output sizes differ");

    double loss = 0.0;

    for(int i = 0; i < predicted.size(); ++i)
    {
        double error = predicted[i] - expected[i];
        loss += 0.5 * error * error;
    }

    return loss;
}

class Neuron {
public:
    vector<double> weights;
    double bias;

    double z;
    double output;

    double delta;

    Neuron(int inputCount)
    {
        weights.resize(inputCount);

        for(int i = 0; i < inputCount; ++i)
            weights[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

        bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

        z = 0.0;
        output = 0.0;
        delta = 0.0;
    }

    double weightedSum(const vector<double>& x)
    {
        if(x.size() != weights.size())
            throw invalid_argument("Input size does not match neuron weights");

        z = bias;

        for(int i = 0; i < weights.size(); ++i)
            z += weights[i] * x[i];

        return z;
    }
};

class Layer {
public:
    vector<Neuron> neurons;
    Activation activation;

    Layer(int neuronCount, int inputCount, Activation a) : activation(a)
    {
        for(int i = 0; i < neuronCount; ++i)
            neurons.emplace_back(inputCount);
    }

    vector<double> calculate(const vector<double>& inputs)
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
};

class NeuralNetwork {
public:
    vector<Layer> layers;

    NeuralNetwork(const vector<Layer>& nnlayers) : layers(nnlayers) {}

    vector<double> calculate(const vector<double>& inputs)
    {
        vector<double> current = inputs;

        for(auto& layer : layers)
            current = layer.calculate(current);

        return current;
    }
};

int main()
{
    NeuralNetwork nn({
        Layer(2, 2, Activation::Sigmoid),
        Layer(2, 2, Activation::Sigmoid),
        Layer(1, 2, Activation::Sigmoid)
    });


    vector<double> input = {1, 0};
    vector<double> output = nn.calculate(input);

    cout << "Output: ";

    for(double value : output)
        cout << value << " ";

    cout << endl;

    return 0;
}