# How to use

```cpp
NeuralNetwork nn({
        Layer(2, 2, Activation::Sigmoid),
        Layer(2, 2, Activation::Sigmoid),
        Layer(1, 2, Activation::Sigmoid)
    });
```

Activations Supported:
- Sigmoid
- Tanh
- ReLU
- Linear

# How to train

```cpp
nn.train(inputs, outputs, epochs, learning_rate, batchtype, batchsize);
```

Add Normalization