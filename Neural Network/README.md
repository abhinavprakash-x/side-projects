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