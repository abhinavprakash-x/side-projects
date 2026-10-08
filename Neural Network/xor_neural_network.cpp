#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

enum class Activation {
    ReLU,
    Sigmoid,
    Tanh
};

enum class BatchType {
    Stochastic,
    Batch,
    MiniBatch
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
    vector<double> weightGradients;
    double biasGradient;

    Neuron(int inputCount)
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

    vector<double> predict(const vector<double>& inputs)
    {
        vector<double> current = inputs;

        for(auto& layer : layers)
            current = layer.calculate(current);

        return current;
    }

    double calculate_loss(const vector<vector<double>>& inputs,
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

    void zero_gradients()
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

    void backward_propagation(const vector<double>& input,
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

    void update_weights(double learningRate, int batchSize)
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

    void train(const vector<vector<double>>& inputs,
               const vector<vector<double>>& expected,
               int epochs, double learningRate,
               BatchType batchType = BatchType::Stochastic,
               int batchSize = 1)
    {
        if(inputs.empty())
            throw invalid_argument("Training dataset is empty");

        if(inputs.size() != expected.size())
            throw invalid_argument("Input and expected dataset sizes differ");

        if(batchType == BatchType::Stochastic) batchSize = 1;
        else if(batchType == BatchType::Batch) batchSize = inputs.size();
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

                update_weights( learningRate, currentBatchSize);
            }

            if(epoch % 1000 == 0)
                cout << "Epoch: " << epoch << " Loss: " << calculate_loss(inputs, expected) << endl;
        }
    }
};

int main()
{
    NeuralNetwork nn({
        Layer(2, 2, Activation::Sigmoid),
        Layer(2, 2, Activation::Sigmoid),
        Layer(1, 2, Activation::Sigmoid)
    });

    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> expected = {{0}, {1}, {1}, {0}};

    nn.train(inputs, expected, 10000, 1.0, BatchType::Stochastic);

    cout << endl;
    cout << "XOR Results:" << endl;

    for(int i = 0; i < inputs.size(); ++i)
    {
        vector<double> output = nn.predict(inputs[i]);
        cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << output[0] << endl;
    }

    return 0;
}