#pragma once

#include <vector>
#include "Layer.h"

using namespace std;

enum class BatchType {
    Stochastic,
    Batch,
    MiniBatch
};

class NeuralNetwork {
public:
    vector<Layer> layers;

    NeuralNetwork(const vector<Layer>& nnlayers);

    vector<double> predict(const vector<double>& inputs);

    double calculate_loss(const vector<vector<double>>& inputs,
                          const vector<vector<double>>& expected);

    void zero_gradients();

    void backward_propagation(const vector<double>& input,
                              const vector<double>& expected);

    void update_weights(double learningRate, int batchSize);

    void train(const vector<vector<double>>& inputs,
               const vector<vector<double>>& expected,
               int epochs, double learningRate,
               BatchType batchType = BatchType::Stochastic,
               int batchSize = 1);
};