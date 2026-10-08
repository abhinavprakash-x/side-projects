#include "Activation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

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
        case Activation::Linear: return z;
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
        case Activation::Linear: return 1.0;
        default: throw invalid_argument("Invalid Activation function");
    }
}