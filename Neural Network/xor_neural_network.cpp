#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

using namespace std;

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