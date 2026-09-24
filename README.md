# Neural Network from Scratch in C++

Beginner Friendly Documentation - https://docs.google.com/document/d/158T4iS1p2u74WOrk20P8EK7UBObdxgY1/edit?usp=drivesdk&ouid=116178795621477601951&rtpof=true&sd=true

Phase 1 : Core Learning - https://docs.google.com/document/d/1ZmWR4USnUDuDr2yMiQS3nKON4PLfna3x/edit?usp=drivesdk&ouid=116178795621477601951&rtpof=true&sd=true

Phase 2 : Implementation & Optimization - https://docs.google.com/document/d/1t3iYNzwKBdsmMwcUAScJgeF1qM-GEhJy/edit?usp=drivesdk&ouid=116178795621477601951&rtpof=true&sd=true

Phase 3 : Interview Mastery - https://docs.google.com/document/d/18B9M6oAVXIGPH2pi8zN6AAYufrmMrChC/edit?usp=drivesdk&ouid=116178795621477601951&rtpof=true&sd=true

A fully connected feedforward neural network implemented from scratch in **C++**, without using external machine-learning frameworks for the core neural-network logic.

The project focuses on understanding how a neural network actually works internally by implementing the major components manually, including:

- Matrix operations
- Forward propagation
- Activation functions
- Softmax
- Cross-entropy loss
- Backpropagation
- Gradient computation
- Stochastic Gradient Descent (SGD)
- MNIST data processing
- Model evaluation
- Numerical debugging
- Memory and performance analysis

The project is designed as an educational implementation to connect the mathematical concepts of neural networks with their actual implementation in C++.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Neural Network Architecture](#neural-network-architecture)
- [Complete Training Pipeline](#complete-training-pipeline)
- [Mathematical Foundations](#mathematical-foundations)
  - [Neuron](#neuron)
  - [Forward Propagation](#forward-propagation)
  - [ReLU](#relu)
  - [Softmax](#softmax)
  - [Cross-Entropy Loss](#cross-entropy-loss)
  - [Backpropagation](#backpropagation)
  - [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [MNIST Dataset](#mnist-dataset)
- [C++ Implementation](#c-implementation)
  - [Matrix Representation](#matrix-representation)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Memory Layout](#memory-layout)
  - [Parameter Initialization](#parameter-initialization)
- [Project Structure](#project-structure)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Convergence Analysis](#convergence-analysis)
- [Memory Efficiency](#memory-efficiency)
- [Performance Analysis](#performance-analysis)
- [Numerical Stability](#numerical-stability)
- [Debugging Strategy](#debugging-strategy)
- [Gradient Checking](#gradient-checking)
- [Training vs Inference](#training-vs-inference)
- [Complexity Analysis](#complexity-analysis)
- [Example Forward Pass](#example-forward-pass)
- [Example Parameter Update](#example-parameter-update)
- [How to Build](#how-to-build)
- [How to Run](#how-to-run)
- [Expected Workflow](#expected-workflow)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [What I Learned](#what-i-learned)
- [Technical Highlights](#technical-highlights)
- [Common Interview Questions](#common-interview-questions)
- [Conclusion](#conclusion)

---

# Project Overview

This project implements a small **feedforward neural network from scratch in C++**.

Instead of using frameworks such as TensorFlow, PyTorch, Keras, or other machine-learning libraries to perform the core training process, the main neural-network operations are implemented directly.

The project demonstrates the complete learning pipeline:

```text
MNIST Image
    ↓
Preprocessing
    ↓
Flatten 28 × 28 Image
    ↓
784 Input Values
    ↓
Dense Layer
    ↓
ReLU Activation
    ↓
Dense Output Layer
    ↓
Softmax
    ↓
Prediction
    ↓
Cross-Entropy Loss
    ↓
Backpropagation
    ↓
Gradients
    ↓
SGD Parameter Update
    ↓
Updated Weights & Biases
```

The main goal is not to build a production-scale deep-learning framework, but to understand the internal mechanics of a neural network and how those mathematical operations translate into actual C++ data structures, loops, memory operations, and numerical calculations.

---

# Objectives

The project was developed with the following objectives:

1. Understand the internal working of a feedforward neural network.
2. Implement forward propagation manually.
3. Implement backpropagation using the chain rule.
4. Implement stochastic gradient descent.
5. Perform matrix operations directly in C++.
6. Train the network using the MNIST dataset.
7. Understand how images are represented numerically.
8. Analyze convergence through training loss.
9. Study memory usage and data layout.
10. Analyze the computational cost of matrix operations.
11. Understand numerical stability issues.
12. Develop systematic debugging techniques for machine-learning code.

---

# Key Features

## Neural Network

- Feedforward neural network
- Fully connected layers
- ReLU activation
- Softmax output
- Cross-entropy loss
- Backpropagation
- Stochastic Gradient Descent

## C++ Implementation

- Custom matrix representation
- Contiguous memory storage
- Matrix multiplication
- Matrix addition
- Transpose operations
- Vector operations
- Manual parameter updates
- Explicit memory-aware implementation

## Dataset

- MNIST handwritten digit dataset
- 28 × 28 grayscale images
- 10 output classes
- Image flattening into numerical vectors

## Analysis

- Loss monitoring
- Accuracy evaluation
- Convergence analysis
- Matrix-operation analysis
- Memory analysis
- Runtime measurement
- Numerical debugging

---

# System Architecture

The project can be viewed as several logical components.

```text
                    ┌────────────────────┐
                    │    MNIST Dataset   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Data Preprocessing │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Input Vector       │
                    │ 784 values         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Dense Layer 1      │
                    │ W1x + b1           │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ ReLU               │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Dense Layer 2      │
                    │ W2a1 + b2          │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Softmax            │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Prediction         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Cross Entropy      │
                    │ Loss               │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Backpropagation    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Gradients          │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ SGD Update         │
                    └────────────────────┘
```

---

# Neural Network Architecture

The reference architecture used throughout this project preparation is:

```text
Input Layer
784 neurons
    ↓
Dense Layer
128 neurons
    ↓
ReLU
    ↓
Dense Layer
10 neurons
    ↓
Softmax
    ↓
Prediction
```

### Why 784 inputs?

MNIST images are:

```text
28 × 28 = 784 pixels
```

Each image can therefore be flattened into a vector containing 784 values.

```text
28 × 28 image
      ↓
Flatten
      ↓
[784 values]
```

### Why 10 outputs?

MNIST contains ten digit classes:

```text
0 1 2 3 4 5 6 7 8 9
```

Therefore, the output layer contains 10 values.

### Why ReLU?

ReLU introduces non-linearity:

```text
ReLU(x) = max(0, x)
```

Without a nonlinear activation function, multiple linear layers would still behave like a single linear transformation.

### Why Softmax?

Softmax converts output logits into values that form a probability distribution across the ten classes.

---

# Complete Training Pipeline

The complete training process is:

```text
1. Load image
       ↓
2. Convert image into numerical vector
       ↓
3. Flatten 28 × 28 → 784
       ↓
4. Forward propagation
       ↓
5. Calculate prediction
       ↓
6. Calculate loss
       ↓
7. Backpropagation
       ↓
8. Calculate gradients
       ↓
9. Update weights and biases using SGD
       ↓
10. Repeat for training samples
       ↓
11. Complete epoch
       ↓
12. Evaluate performance
```

The complete conceptual loop is:

```text
Input
 ↓
Forward Pass
 ↓
Prediction
 ↓
Loss
 ↓
Backward Pass
 ↓
Gradients
 ↓
SGD
 ↓
Updated Parameters
 ↓
Next Sample
```

---

# Mathematical Foundations

## Neuron

A basic neuron calculates:

\[
z = \sum_i w_i x_i + b
\]

where:

- \(x_i\) = input
- \(w_i\) = weight
- \(b\) = bias
- \(z\) = weighted sum

An activation function is then applied:

\[
a = f(z)
\]

---

## Forward Propagation

For the reference architecture:

### First layer

\[
z_1 = W_1x + b_1
\]

### ReLU

\[
a_1 = ReLU(z_1)
\]

### Second layer

\[
z_2 = W_2a_1 + b_2
\]

### Softmax

\[
\hat{y} = Softmax(z_2)
\]

The output \(\hat{y}\) contains the predicted class probabilities.

---

# ReLU

The Rectified Linear Unit is:

\[
ReLU(x) = \max(0,x)
\]

Example:

```text
x = -5 → 0
x = -1 → 0
x =  0 → 0
x =  2 → 2
x =  7 → 7
```

Its derivative is:

\[
ReLU'(x) =
\begin{cases}
1 & x > 0 \\
0 & x \leq 0
\end{cases}
\]

ReLU is used in the hidden layer to introduce non-linearity.

---

# Softmax

For logits \(z_1, z_2, ..., z_n\):

\[
Softmax(z_i)=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

The outputs sum approximately to 1.

Example:

```text
Class 0 → 0.01
Class 1 → 0.02
Class 2 → 0.03
Class 3 → 0.01
Class 4 → 0.05
Class 5 → 0.03
Class 6 → 0.02
Class 7 → 0.80
Class 8 → 0.02
Class 9 → 0.01
```

The prediction is the class with the largest probability.

```text
Prediction = argmax(probabilities)
```

For the example above:

```text
Prediction = 7
```

---

# Cross-Entropy Loss

For a classification problem with one-hot target \(y\):

\[
L = -\sum_i y_i \log(\hat{y_i})
\]

For a single correct class:

\[
L = -\log(\hat{y}_{correct})
\]

Therefore:

```text
High probability for correct class
        ↓
Lower loss

Low probability for correct class
        ↓
Higher loss
```

Cross-entropy measures how different the predicted probability distribution is from the target distribution.

---

# Backpropagation

Backpropagation is the process used to calculate the gradients of the loss with respect to the model parameters.

The main idea is:

```text
Loss
 ↓
Output Layer
 ↓
Hidden Layer
 ↓
Input Layer
```

The chain rule is used to propagate the effect of the loss backward through the network.

For the output layer:

\[
\delta_2 = \hat{y} - y
\]

Then:

\[
dW_2 = \delta_2 a_1^T
\]

\[
db_2 = \delta_2
\]

For the hidden layer:

\[
\delta_1 =
(W_2^T\delta_2)
\odot ReLU'(z_1)
\]

Then:

\[
dW_1 = \delta_1 x^T
\]

\[
db_1 = \delta_1
\]

The gradients are then given to the optimizer.

---

# Stochastic Gradient Descent

SGD updates the parameters using:

\[
\theta_{new}
=
\theta_{old}
-
\eta \nabla L
\]

For weights:

\[
W_{new}=W_{old}-\eta dW
\]

For biases:

\[
b_{new}=b_{old}-\eta db
\]

where:

- \(W\) = weights
- \(b\) = biases
- \(\eta\) = learning rate
- \(dW\) = gradient of weights
- \(db\) = gradient of biases

The negative sign moves the parameters in the direction that reduces the loss.

---

# MNIST Dataset

MNIST is a handwritten digit dataset commonly used for classification experiments.

Each image contains:

```text
28 × 28 pixels
```

Therefore:

```text
28 × 28 = 784 values
```

Each sample contains:

```text
Image + Label
```

Example:

```text
Image → handwritten digit
Label → 7
```

---

## Image Representation

The original image is two-dimensional:

```text
28 × 28
```

For a fully connected network, it can be flattened:

```text
28 × 28
     ↓
784-element vector
```

Conceptually:

```text
[
 pixel1,
 pixel2,
 pixel3,
 ...
 pixel784
]
```

---

## Output Representation

The output layer contains 10 values:

```text
[0,1,2,3,4,5,6,7,8,9]
```

A one-hot target for digit 7 can be represented as:

```text
[0,0,0,0,0,0,0,1,0,0]
```

---

## MNIST Subset

The project uses an MNIST subset for a manageable educational training experiment.

The exact subset size should match the dataset configuration used in the implementation.

> **Note:** Do not claim a specific sample count in the README unless that number is actually used by the project.

---

# C++ Implementation

The neural network is implemented using standard C++ data structures and numerical operations.

The implementation focuses on:

- Matrix representation
- Matrix multiplication
- Matrix addition
- Transpose
- Activation functions
- Gradient calculations
- Parameter updates
- Memory management

---

# Matrix Representation

A simple custom matrix class can store:

```cpp
class Matrix {
private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;

public:
    Matrix(size_t rows, size_t cols)
        : rows_(rows),
          cols_(cols),
          data_(rows * cols, 0.0) {}

    double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    const double& operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    size_t rows() const {
        return rows_;
    }

    size_t cols() const {
        return cols_;
    }
};
```

This representation uses one contiguous `std::vector`.

---

# Matrix Multiplication

For:

```text
A = m × n
B = n × p
```

the result is:

```text
C = m × p
```

The mathematical operation is:

\[
C_{ij}=\sum_k A_{ik}B_{kj}
\]

A direct C++ implementation can be:

```cpp
Matrix multiply(const Matrix& A, const Matrix& B)
{
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    Matrix C(A.rows(), B.cols());

    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {

            double sum = 0.0;

            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }

            C(i, j) = sum;
        }
    }

    return C;
}
```

The three loops correspond to:

```text
Row
 ↓
Column
 ↓
Dot Product
```

---

# Memory Layout

Matrices can be stored in a one-dimensional array using row-major indexing:

\[
index = row \times columns + column
\]

For example:

```text
Matrix:

1 2 3
4 5 6

Flat storage:

[1, 2, 3, 4, 5, 6]
```

The index for:

```text
row = 1
column = 2
```

is:

```text
1 × 3 + 2 = 5
```

Therefore:

```text
data[5] = 6
```

---

## Why Contiguous Storage?

Contiguous storage provides:

- Simple indexing
- Predictable memory layout
- Better sequential memory access
- Efficient interaction with CPU caches
- Fewer separate allocations compared with nested dynamic structures

This does not automatically guarantee a faster implementation, but it provides a practical memory layout for numerical operations.

---

# Parameter Initialization

Neural-network parameters should not all start with identical values.

A common approach is to initialize weights using small random values.

Example:

```cpp
std::random_device rd;
std::mt19937 generator(rd());

std::normal_distribution<double> distribution(0.0, 0.01);

double weight = distribution(generator);
```

The exact initialization strategy should match the implementation.

### Why random initialization?

If every neuron starts with exactly the same parameters, the neurons can remain symmetric and learn similar representations.

---

# Project Structure

A possible organization for the project is:

```text
neural-network-from-scratch/
│
├── include/
│   ├── Matrix.h
│   ├── Layer.h
│   ├── NeuralNetwork.h
│   └── Activations.h
│
├── src/
│   ├── Matrix.cpp
│   ├── Layer.cpp
│   ├── NeuralNetwork.cpp
│   └── main.cpp
│
├── data/
│   └── mnist/
│
├── results/
│   ├── loss.csv
│   └── metrics.txt
│
├── CMakeLists.txt
│
└── README.md
```

The exact organization may differ depending on the final implementation.

---

# Training Process

A simplified training loop is:

```cpp
for (int epoch = 0; epoch < epochs; ++epoch)
{
    for (const auto& sample : trainingData)
    {
        // Forward propagation
        auto prediction = network.forward(sample.input);

        // Calculate loss
        double loss =
            network.loss(prediction, sample.label);

        // Backpropagation
        network.backward(
            sample.input,
            sample.label
        );

        // Update parameters
        network.update(learningRate);
    }
}
```

The important sequence is:

```text
Forward
   ↓
Loss
   ↓
Backward
   ↓
Gradients
   ↓
Update
```

---

# Evaluation

After training, the network can be evaluated on samples that were not used for parameter updates.

Prediction:

```cpp
auto output = network.forward(input);

int predictedClass = argmax(output);
```

Accuracy is:

\[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\]

For example:

```text
Correct predictions = 850
Total predictions   = 1000

Accuracy = 85%
```

The actual project accuracy should be reported only from the implementation's measured results.

---

# Convergence Analysis

Convergence refers to the behavior of the optimization process as training progresses.

A common signal is the training loss.

Example conceptual behavior:

```text
Loss
 ^
 |\
 | \
 |  \
 |   \
 |    \____
 |         \___
 |
 +--------------------> Epoch
```

A decreasing loss generally indicates that the optimization process is improving the model with respect to the selected objective.

However:

```text
Lower loss ≠ automatically perfect accuracy
```

Accuracy and loss measure different aspects of the model.

---

# Memory Efficiency

The project also examines how model parameters and intermediate values are represented in memory.

For the reference architecture:

```text
784 → 128 → 10
```

### First layer

```text
128 × 784
= 100,352 weights
```

### Second layer

```text
10 × 128
= 1,280 weights
```

### Biases

```text
128 + 10
= 138
```

### Total parameters

```text
100,352
+ 1,280
+ 138
----------------
101,770 parameters
```

If stored using `double`:

```text
101,770 × 8 bytes
≈ 814,160 bytes
≈ 795 KiB
```

If stored using `float`:

```text
101,770 × 4 bytes
≈ 407,080 bytes
≈ 397 KiB
```

These numbers represent parameter storage only.

Actual memory consumption also includes:

- Activations
- Gradients
- Input samples
- Temporary matrices
- Dataset storage
- Objects and program overhead

---

# Performance Analysis

Matrix multiplication is one of the major computational components of this network.

For:

```text
A = m × n
B = n × p
```

standard matrix multiplication has:

\[
O(mnp)
\]

time complexity.

For square matrices:

\[
O(n^3)
\]

The project therefore considers:

- Matrix multiplication cost
- Memory access patterns
- Temporary allocations
- Object copies
- Contiguous storage
- Loop structure
- Cache behavior

---

# Measuring Performance

Performance should be measured rather than assumed.

A simple timing approach in C++ is:

```cpp
#include <chrono>

auto start =
    std::chrono::high_resolution_clock::now();

// Operation being measured

auto end =
    std::chrono::high_resolution_clock::now();

auto duration =
    std::chrono::duration_cast<
        std::chrono::microseconds
    >(end - start);

std::cout
    << duration.count()
    << " microseconds\n";
```

A more reliable benchmark should:

1. Run multiple iterations.
2. Use consistent input sizes.
3. Avoid unrelated work during measurement.
4. Compare equivalent implementations.
5. Measure before and after optimization.

---

# Numerical Stability

Numerical stability is particularly important when implementing neural networks manually.

One common issue is Softmax.

A naive implementation:

```text
exp(z[i])
```

can overflow when the logits are very large.

A more stable approach subtracts the maximum logit first:

\[
Softmax(z_i)=
\frac{e^{z_i-\max(z)}}
{\sum_j e^{z_j-\max(z)}}
\]

Because the same value is subtracted from every logit, the resulting probability distribution remains unchanged while numerical behavior is improved.

---

# Debugging Strategy

When the network does not train correctly, debugging should be systematic.

A useful process is:

```text
Problem
   ↓
Reproduce
   ↓
Isolate
   ↓
Inspect
   ↓
Test Hypothesis
   ↓
Fix
   ↓
Retest
```

---

## Check 1: Input Data

Verify:

- Input shape
- Pixel range
- Flattening
- Labels
- Data type
- Sample/label alignment

---

## Check 2: Matrix Dimensions

For the reference architecture:

```text
x  = 784 × 1

W1 = 128 × 784
b1 = 128 × 1

W2 = 10 × 128
b2 = 10 × 1
```

Therefore:

```text
W1 × x
= (128 × 784)(784 × 1)
= 128 × 1
```

Then:

```text
W2 × a1
= (10 × 128)(128 × 1)
= 10 × 1
```

Dimension checking can catch many bugs early.

---

## Check 3: Forward Pass

Verify each intermediate value:

```text
x
 ↓
z1
 ↓
a1
 ↓
z2
 ↓
softmax
```

Check for:

- Unexpected zeros
- Extremely large values
- NaN
- Infinity
- Incorrect dimensions

---

## Check 4: Loss

If the loss is:

```text
NaN
```

possible causes include:

- `log(0)`
- Numerical overflow
- Invalid input
- Extremely large parameter values

---

## Check 5: Gradients

Check whether gradients are:

- Finite
- Non-zero when expected
- Reasonable in magnitude

A gradient that is always zero may indicate a bug in the backward pass or activation derivative.

---

## Check 6: Parameter Updates

Verify that the parameters actually change.

For example:

```text
Before update:
W[0] = 0.0123

After update:
W[0] = 0.0118
```

If weights never change, the training process is not actually updating the model.

---

# Gradient Checking

Gradient checking is useful for validating backpropagation.

The numerical approximation for a parameter \(w\) is:

\[
\frac{\partial L}{\partial w}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}
{2\epsilon}
\]

This can be compared against the analytically calculated gradient.

Conceptually:

```text
Analytical Gradient
        vs
Numerical Gradient
```

If they are sufficiently close, the derivative implementation is more likely to be correct.

Gradient checking is generally used on small test cases because numerical evaluation requires additional forward passes.

---

# Training vs Inference

## Training

Training includes:

```text
Input
 ↓
Forward Pass
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient Calculation
 ↓
Parameter Update
```

---

## Inference

Inference only needs:

```text
Input
 ↓
Forward Pass
 ↓
Prediction
```

There is no:

```text
Backpropagation
Gradient Calculation
Parameter Update
```

during inference.

---

# Complexity Analysis

For matrix multiplication:

```text
A = m × n
B = n × p
```

the standard algorithm is:

\[
O(mnp)
\]

For the first dense layer:

```text
128 × 784
```

The number of multiply-accumulate operations is proportional to:

```text
128 × 784
```

For the second layer:

```text
10 × 128
```

Therefore, the first dense layer represents a larger portion of the basic matrix multiplication workload for this architecture.

---

# Example Forward Pass

Suppose:

```text
Input:
x = 784 × 1
```

First layer:

```text
W1 = 128 × 784
b1 = 128 × 1
```

Calculate:

\[
z_1=W_1x+b_1
\]

Then:

\[
a_1=ReLU(z_1)
\]

Second layer:

```text
W2 = 10 × 128
b2 = 10 × 1
```

Calculate:

\[
z_2=W_2a_1+b_2
\]

Then:

\[
\hat{y}=Softmax(z_2)
\]

Finally:

```text
prediction = argmax(y_hat)
```

---

# Example Parameter Update

Suppose:

```text
Weight:
w = 0.50

Gradient:
dw = 0.20

Learning rate:
η = 0.01
```

The update is:

\[
w_{new}
=
w-\eta dw
\]

Therefore:

```text
w_new
= 0.50 - (0.01 × 0.20)
= 0.498
```

If the gradient were negative:

```text
w     = 0.50
dw    = -0.20
η     = 0.01
```

then:

```text
w_new
= 0.50 - (0.01 × -0.20)
= 0.502
```

---

# How to Build

## Prerequisites

Make sure the system has:

- C++ compiler with modern C++ support
- CMake (if using CMake)
- MNIST dataset files
- Standard C++ library

Examples of supported compilers include:

```text
GCC
Clang
MSVC
```

---

## Build Using CMake

Create a build directory:

```bash
mkdir build
cd build
```

Generate build files:

```bash
cmake ..
```

Build the project:

```bash
cmake --build .
```

For a release build:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Use the exact command appropriate to the project's final build configuration.

---

# How to Run

After compilation, run the generated executable.

Example:

```bash
./neural_network
```

On Windows, depending on the generator:

```powershell
.\Release\neural_network.exe
```

The program should perform the configured workflow:

```text
Load Dataset
    ↓
Preprocess Data
    ↓
Initialize Network
    ↓
Train
    ↓
Track Loss
    ↓
Evaluate
    ↓
Display Results
```

---

# Expected Workflow

A typical execution looks conceptually like:

```text
Loading MNIST...
Dataset loaded.

Initializing network...
Network initialized.

Training...
Epoch 1
Epoch 2
Epoch 3
...

Training complete.

Evaluating model...
Accuracy: <measured value>

Saving metrics...
```

The exact console output depends on the implementation.

---

# Results

Results should be filled from actual experiments rather than estimated values.

Recommended metrics to record:

| Metric | Value |
|---|---|
| Training samples | `<actual value>` |
| Test samples | `<actual value>` |
| Epochs | `<actual value>` |
| Learning rate | `<actual value>` |
| Batch size | `<actual value>` |
| Final training loss | `<actual value>` |
| Evaluation accuracy | `<actual value>` |
| Average training time | `<actual value>` |
| Parameter count | 101,770 for 784→128→10 |
| Data type | `<float/double>` |

### Example Result Format

```text
Architecture:
784 → 128 → 10

Activation:
ReLU

Output:
Softmax

Loss:
Cross Entropy

Optimizer:
SGD

Training:
MNIST subset

Final Loss:
<measured value>

Accuracy:
<measured value>
```

> Do not add fabricated accuracy, latency, convergence, or optimization percentages to this section.

---

# Limitations

This project is intentionally a small educational implementation.

Some limitations include:

- Fully connected architecture instead of a CNN
- Basic SGD optimization
- Limited scalability compared with specialized ML frameworks
- Manual matrix operations
- No GPU acceleration in the core implementation unless explicitly added
- Limited abstraction compared with production ML libraries
- MNIST is a relatively simple dataset
- Performance is dependent on implementation and hardware
- Numerical and memory optimizations are limited compared with optimized libraries

The purpose is understanding rather than replacing established deep-learning frameworks.

---

# Future Improvements

Potential improvements include:

## Optimization

- Mini-batch training
- Momentum
- Adam optimizer
- Better weight initialization
- Matrix multiplication optimization
- Buffer reuse
- SIMD/vectorization
- Parallelization
- Multithreading

## Architecture

- Multiple hidden layers
- Dropout
- Batch normalization
- Different activation functions
- Convolutional neural networks

## Dataset

- Full MNIST dataset
- CIFAR datasets
- Custom image datasets

## Engineering

- Better configuration management
- Logging
- Unit tests
- Automated tests
- Benchmarking framework
- More robust data loaders
- Model serialization

---

# What I Learned

This project helped connect neural-network theory with low-level implementation.

The major learning areas were:

### Machine Learning

- Neural-network fundamentals
- Forward propagation
- Backpropagation
- Chain rule
- Gradient descent
- Loss functions
- Classification
- Model evaluation

### C++

- Classes and encapsulation
- `std::vector`
- References
- `const`
- Memory layout
- Dynamic storage
- Exception handling
- Numerical loops
- Random number generation
- Timing

### Systems and Performance

- Contiguous memory
- Cache locality
- Matrix computation
- Allocation overhead
- Computational complexity
- Runtime measurement
- Memory usage

### Debugging

- Dimension checking
- Numerical stability
- NaN detection
- Gradient validation
- Training-loop debugging
- Isolating mathematical vs implementation errors

---

# Technical Highlights

The project demonstrates understanding of:

```text
Neural Network
      ↓
Mathematics
      ↓
Matrix Operations
      ↓
C++ Implementation
      ↓
Memory Management
      ↓
Performance Analysis
      ↓
Debugging
```

Important concepts implemented or analyzed include:

- Feedforward computation
- ReLU activation
- Softmax
- Cross-entropy
- Backpropagation
- SGD
- Matrix multiplication
- Gradient calculation
- Parameter updates
- MNIST preprocessing
- Accuracy evaluation
- Numerical stability
- Memory layout
- Runtime measurement

---

# Common Interview Questions

## What does "from scratch" mean in this project?

It means the core neural-network computations were implemented directly in C++ instead of delegating the model training process to frameworks such as TensorFlow or PyTorch.

---

## Why did you choose C++?

C++ provides direct control over:

- Data structures
- Memory representation
- Loops
- Numerical operations
- Performance characteristics

It is therefore useful for understanding how machine-learning computations are translated into low-level implementation details.

---

## Why is MNIST a good dataset for this project?

MNIST is relatively simple and well structured.

Each image has:

```text
28 × 28
```

pixels and belongs to one of:

```text
10 classes
```

This makes it suitable for understanding the complete neural-network training pipeline without introducing excessive dataset complexity.

---

## What is the difference between forward propagation and backpropagation?

Forward propagation calculates the prediction.

Backpropagation calculates how the loss changes with respect to the model parameters.

```text
Forward:
Input → Prediction

Backward:
Loss → Gradients
```

---

## Where do the gradients come from?

The gradients are calculated during backpropagation using the chain rule.

---

## What does SGD do?

SGD uses the gradients to update the weights and biases:

\[
\theta_{new}
=
\theta_{old}
-
\eta\nabla L
\]

---

## What happens if the learning rate is too high?

The optimization can overshoot useful parameter values, causing unstable training or increasing loss.

---

## What happens if the learning rate is too low?

Training can become unnecessarily slow.

---

## Why use ReLU?

ReLU provides a nonlinear transformation while being computationally simple.

---

## Why use Softmax in the output layer?

Because the problem is a multi-class classification problem with ten mutually exclusive classes.

---

## Why use cross-entropy?

Cross-entropy is suitable for classification with probability outputs and measures how well the predicted distribution matches the target distribution.

---

## Why use a custom matrix class?

The custom matrix representation allows the project to explicitly control:

- Storage
- Indexing
- Dimensions
- Multiplication
- Addition
- Transpose
- Memory layout

---

## Why use `std::vector`?

`std::vector` provides dynamically sized contiguous storage and manages memory automatically.

---

## What is the complexity of matrix multiplication?

For:

```text
A = m × n
B = n × p
```

the standard algorithm is:

\[
O(mnp)
\]

---

## How did you analyze performance?

The main areas considered were:

- Matrix multiplication
- Memory access
- Repeated allocations
- Copies
- Data layout
- Loop structure

Runtime should be measured before making quantitative performance claims.

---

## How did you analyze memory efficiency?

The analysis considered:

- Weight storage
- Activation storage
- Gradient storage
- Temporary matrices
- Contiguous representation
- Avoiding unnecessary copies
- Repeated allocations

---

# Conclusion

This project demonstrates how a neural network can be implemented using fundamental mathematical operations and standard C++ data structures.

Instead of treating neural networks as a black box, the project follows the complete process:

```text
MNIST
  ↓
Data Representation
  ↓
Matrix Operations
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
SGD
  ↓
Updated Parameters
  ↓
Evaluation
```

The project provides a practical understanding of how concepts such as:

```text
Weights
Biases
Activations
Matrix Multiplication
Loss
Gradients
Backpropagation
Optimization
Memory
Performance
```

work together to train a machine-learning model.

The key objective is not simply to obtain a prediction, but to understand the complete path from mathematical equations to executable C++ code.

---

# Final Project Summary

```text
Project:
Neural Network from Scratch

Language:
C++

Core Concepts:
Forward Propagation
Backpropagation
SGD
Matrix Operations
Softmax
Cross Entropy
ReLU

Dataset:
MNIST

Input:
28 × 28 → 784 values

Reference Architecture:
784 → 128 → 10

Output:
10-class digit classification

Implementation Focus:
C++ Data Structures
Memory Layout
Matrix Computation
Performance Analysis
Debugging

Main Goal:
Understand and implement the internal mechanics
of a neural network without relying on an external
machine-learning framework for the core training logic.
```

---

## Author

**Sanket Motewar**

B.Tech — Computer Science & Artificial Intelligence  
Vishwakarma Institute of Technology, Pune