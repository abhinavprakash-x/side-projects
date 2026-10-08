#pragma once

#include <vector>
using namespace std;

class Neuron {
public:
    vector<double> weights;
    double bias;

    double z;
    double output;

    double delta;
    vector<double> weightGradients;
    double biasGradient;

    Neuron(int inputCount);

    double weightedSum(const vector<double>& x);
};