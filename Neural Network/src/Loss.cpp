#include "Loss.h"
#include <stdexcept>
using namespace std;

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