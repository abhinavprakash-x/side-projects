#pragma once

enum class Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear
};

double ReLU(double z);
double sigmoid(double z);
double Tanh(double z);

double activate(double z, Activation type);
double activationDerivative(double z, Activation type);