# Mathematical Explanation of the 3-Layer Neural Network

This document provides a detailed, step-by-step tutorial on the mathematical derivations and calculations behind the questions for this neural network problem.

---

## 1. Network Architecture Overview

The neural network consists of:
*   **Input layer**: Input vector x (size 3)
*   **Hidden Layer 1**: Pre-activation a1 (size 3), Post-activation h1 (size 3) using Sigmoid
*   **Hidden Layer 2**: Pre-activation a2 (size 3), Post-activation h2 (size 3) using Sigmoid
*   **Output Layer**: Pre-activation a3 (size 3), Output predictions y_hat = O (size 3) using Softmax

### Inputs and Target
*   **Input vector**: 
    ```text
    x = [ 1 ]
        [ 0 ]
        [ 1 ]
    ```
*   **Target labels**: 
    ```text
    y = [ 0 ]
        [ 0 ]
        [ 1 ]
    ```
*   **Activation functions**:
    *   **Sigmoid**: `sigma(z) = 1 / (1 + exp(-z))`
    *   **Softmax**: `softmax(z)_i = exp(z_i) / sum_j(exp(z_j))`
*   **Loss function**: Cross-Entropy Loss (using natural log)
    `L = -sum_i (y_i * ln(y_hat_i))`

---

## 2. Question 2: Pre-activation Vector a1

### Mathematical Formula
The pre-activation vector of the first hidden layer is obtained by multiplying the weight matrix W1 by the input vector x and adding the bias vector b1:
`a1 = W1 * x + b1`

### Calculation
Given:
```text
W1 = [ 0.5488135   0.71518937  0.60276338 ]
     [ 0.54488318  0.4236548   0.64589411 ]
     [ 0.43758721  0.891773    0.96366276 ]

b1 = [ 0.38344152 ]
     [ 0.79172504 ]
     [ 0.52889492 ]
```

Multiplying by x = [1, 0, 1]^T:
```text
W1 * x = [ 0.5488135*1  + 0.71518937*0 + 0.60276338*1 ]   [ 1.15157688 ]
         [ 0.54488318*1 + 0.4236548*0  + 0.64589411*1 ] = [ 1.19077729 ]
         [ 0.43758721*1 + 0.891773*0   + 0.96366276*1 ]   [ 1.40124997 ]
```

Adding bias b1:
```text
a1 = [ 1.15157688 ]   [ 0.38344152 ]   [ 1.5350184  ]
     [ 1.19077729 ] + [ 0.79172504 ] = [ 1.98250233 ]
     [ 1.40124997 ]   [ 0.52889492 ]   [ 1.93014489 ]
```

### Sum of elements of a1
`Sum(a1) = 1.5350184 + 1.98250233 + 1.93014489 = 5.4477`

*   **Answer**: **Option B (5.44)**

---

## 3. Question 3: Post-activation Vector h1

### Mathematical Formula
The post-activation vector is the element-wise sigmoid activation of the pre-activation vector:
`h1 = sigma(a1) = 1 / (1 + exp(-a1))`

### Calculation
Using the elements of a1:
1.  `h1_1 = 1 / (1 + exp(-1.5350184))  = 0.822739`
2.  `h1_2 = 1 / (1 + exp(-1.98250233)) = 0.878948`
3.  `h1_3 = 1 / (1 + exp(-1.93014489)) = 0.873265`

Thus:
```text
h1 = [ 0.82273938 ]
     [ 0.87894766 ]
     [ 0.87326546 ]
```

### Sum of elements of h1
`Sum(h1) = 0.82273938 + 0.87894766 + 0.87326546 = 2.57495`

*   **Answer**: **Option A (2.57)**

---

## 4. Question 4: Forward Pass and Loss

To calculate the loss, we first complete the forward propagation.

### Hidden Layer 2 Calculations
1.  **Pre-activations**: `a2 = W2 * h1 + b2`
    Using W2 and b2 from the parameters dump:
    ```text
    a2 = [ 2.10091008 ]  => Sum(a2) = 6.46
         [ 1.76019958 ]
         [ 2.59905711 ]
    ```
2.  **Post-activations**: `h2 = sigma(a2)`
    ```text
    h2 = [ 0.89099238 ]  => Sum(h2) = 2.63
         [ 0.85324546 ]
         [ 0.9308477  ]
    ```

### Output Layer Calculations
1.  **Pre-activations**: `a3 = W3 * h2 + b3`
    ```text
    a3 = [ 1.34960588 ]  => Sum(a3) = 4.87
         [ 1.70564619 ]
         [ 1.81966891 ]
    ```
2.  **Softmax predictions**: `y_hat = softmax(a3)`
    `y_hat_i = exp(a3_i) / (exp(a3_1) + exp(a3_2) + exp(a3_3))`
    
    *   `exp(a3_1) = exp(1.34960588) = 3.8559`
    *   `exp(a3_2) = exp(1.70564619) = 5.5050`
    *   `exp(a3_3) = exp(1.81966891) = 6.1700`
    *   Sum of exponents = 3.8559 + 5.5050 + 6.1700 = 15.5309
    
    Dividing each exponent by the sum:
    ```text
    y_hat = [ 0.23691422 ]
            [ 0.33838847 ]
            [ 0.42469732 ]
    ```

### Loss Calculation (Cross-Entropy)
Since the label is one-hot encoded as `y = [0, 0, 1]^T`, the loss simplifies to:
`L = -ln(y_hat_3) = -ln(0.42469732) = 0.856379`

*   **Answer**: **`0.856379`** (rounds to **0.86**)

---

## 5. Question 5: Gradient with respect to Output Pre-activations (grad_a3)

### Mathematical Derivation
We want to find the partial derivatives of the cross-entropy loss L with respect to the output pre-activations a3_i.
For a true label class k where y_k = 1, it can be proven that the gradient of Softmax + Cross Entropy simplifies beautifully:
`dL / da3_i = y_hat_i - y_i`

In vector form:
`grad_a3 = y_hat - y`

### Calculation
Given:
```text
y_hat = [ 0.23691422 ]     y = [ 0 ]
        [ 0.33838847 ]         [ 0 ]
        [ 0.42469732 ]         [ 1 ]
```

```text
grad_a3 = [ 0.23691422 ]   [ 0 ]   [  0.23691422 ]
          [ 0.33838847 ] - [ 0 ] = [  0.33838847 ]
          [ 0.42469732 ]   [ 1 ]   [ -0.57530268 ]
```

*   **Answer**: **Option E** (`[ 0.23,  0.33, -0.57 ]^T`)

---

## 6. Question 6: Gradient with respect to Bias (grad_b2)

### Mathematical Derivation (Backpropagating through layer 2)
We want to find the gradient of the loss with respect to the bias of the second hidden layer, b2.
Using the chain rule:
`grad_b2 = grad_a2 * (da2 / db2)`

Since `a2 = W2 * h1 + b2`, we have `da2 / db2 = I` (the identity matrix). Hence:
`grad_b2 = grad_a2`

Let's find `grad_a2`:
`grad_a2 = grad_h2 * sigma'(a2)` (element-wise)

Where:
1.  **Gradient backpropagated from layer 3**:
    Since `a3 = W3 * h2 + b3`, the derivative with respect to h2 is:
    `grad_h2 = W3^T * grad_a3`
2.  **Sigmoid Derivative**:
    `sigma'(a2) = h2 * (1 - h2)` (element-wise)

Combining these:
`grad_b2 = (W3^T * (y_hat - y)) * h2 * (1 - h2)` (where `*` between vectors here denotes element-wise multiplication)

### Calculation
Using the computed values:
1.  `W3^T * (y_hat - y)` 
    ```text
    = [ 0.11827  0.94467  0.26456 ]   [  0.23691 ]   [  0.19535 ]
      [ 0.63992  0.52185  0.77423 ] * [  0.33839 ] = [ -0.11718 ]
      [ 0.14335  0.41466  0.45615 ]   [ -0.57530 ]   [ -0.08846 ]
    ```
2.  `h2 * (1 - h2)`
    ```text
    = [ 0.89099 * 0.10901 ]   [ 0.09712 ]
      [ 0.85325 * 0.14675 ] = [ 0.12522 ]
      [ 0.93085 * 0.06915 ]   [ 0.06437 ]
    ```
3.  `grad_b2`
    ```text
    = [  0.19535 ]   [ 0.09712 ]   [  0.01838 ]
      [ -0.11718 ] * [ 0.12522 ] = [ -0.01998 ]
      [ -0.08846 ]   [ 0.06437 ]   [ -0.00384 ]
    ```

*   **Answer**: **Option A** (`[ 0.018, -0.019, -0.003 ]^T`)

---

## 7. Question 7: Gradient Updates and New Loss (eta = 1)

We calculate the gradients for all weights and biases in the network and update them using gradient descent:
`theta_new = theta - eta * grad_theta`

### Gradient Formulas for All Parameters
*   **Layer 3**:
    `grad_W3 = grad_a3 * h2^T`
    `grad_b3 = grad_a3`
*   **Layer 2**:
    `grad_W2 = grad_a2 * h1^T`
    `grad_b2 = grad_a2`
*   **Layer 1**:
    `grad_a1 = (W2^T * grad_a2) * h1 * (1 - h1)`
    `grad_W1 = grad_a1 * x^T`
    `grad_b1 = grad_a1`

### Parameter Updates (eta = 1)
Subtract the computed gradient from each corresponding parameter matrix:
*   `W_layer_new = W_layer - grad_W_layer`
*   `b_layer_new = b_layer - grad_b_layer`

### New Forward Pass and Loss
Propagating the input `x = [1, 0, 1]^T` through the network with updated parameters yields:
1.  **Layer 1 new**: 
    ```text
    a1_new = [ 1.62479 ]  =>  h1_new = [ 0.83546 ]
             [ 2.08027 ]               [ 0.88898 ]
             [ 2.01529 ]               [ 0.88237 ]
    ```
2.  **Layer 2 new**: 
    ```text
    a2_new = [ 2.11589 ]  =>  h2_new = [ 0.89246 ]
             [ 1.74542 ]               [ 0.85137 ]
             [ 2.59560 ]               [ 0.93060 ]
    ```
3.  **Layer 3 new**: 
    ```text
    a3_new = [ 1.13854 ]  =>  y_hat_new = [ 0.048383 ]
             [ 1.41743 ]                  [ 0.063943 ]
             [ 4.09204 ]                  [ 0.897674 ]
    ```

### New Cross-Entropy Loss
`L_new = -ln(y_hat_new_3) = -ln(0.897674) = 0.072533`

*   **Answer**: **`0.072533`** (rounds to **0.07**)
