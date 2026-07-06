# Mathematical Explanation of the 3-Layer Neural Network

This document provides a detailed, step-by-step tutorial on the mathematical derivations and calculations behind the questions for this neural network problem.

---

## 1. Network Architecture Overview

The neural network consists of:
*   **Input layer**: Input vector $x$ (size 3)
*   **Hidden Layer 1**: Pre-activation $a_1$ (size 3), Post-activation $h_1$ (size 3) using Sigmoid
*   **Hidden Layer 2**: Pre-activation $a_2$ (size 3), Post-activation $h_2$ (size 3) using Sigmoid
*   **Output Layer**: Pre-activation $a_3$ (size 3), Output predictions $\hat{y} = O$ (size 3) using Softmax

### Inputs and Target
*   **Input vector**: 
    $$x = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$
*   **Target labels**: 
    $$y = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$
*   **Activation functions**:
    *   **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
    *   **Softmax**: $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
*   **Loss function**: Cross-Entropy Loss (using natural log)
    $$L = -\sum_i (y_i \ln(\hat{y}_i))$$

---

## 2. Question 2: Pre-activation Vector $a_1$

### Mathematical Formula
The pre-activation vector of the first hidden layer is obtained by multiplying the weight matrix $W_1$ by the input vector $x$ and adding the bias vector $b_1$:
$$a_1 = W_1 x + b_1$$

### Calculation
Given:
$$W_1 = \begin{bmatrix} 0.5488135 & 0.71518937 & 0.60276338 \\ 0.54488318 & 0.4236548 & 0.64589411 \\ 0.43758721 & 0.891773 & 0.96366276 \end{bmatrix}$$

$$b_1 = \begin{bmatrix} 0.38344152 \\ 0.79172504 \\ 0.52889492 \end{bmatrix}$$

Multiplying by $x = [1, 0, 1]^T$:
$$W_1 x = \begin{bmatrix} 0.5488135 \cdot 1 + 0.71518937 \cdot 0 + 0.60276338 \cdot 1 \\ 0.54488318 \cdot 1 + 0.4236548 \cdot 0 + 0.64589411 \cdot 1 \\ 0.43758721 \cdot 1 + 0.891773 \cdot 0 + 0.96366276 \cdot 1 \end{bmatrix} = \begin{bmatrix} 1.15157688 \\ 1.19077729 \\ 1.40124997 \end{bmatrix}$$

Adding bias $b_1$:
$$a_1 = \begin{bmatrix} 1.15157688 \\ 1.19077729 \\ 1.40124997 \end{bmatrix} + \begin{bmatrix} 0.38344152 \\ 0.79172504 \\ 0.52889492 \end{bmatrix} = \begin{bmatrix} 1.5350184 \\ 1.98250233 \\ 1.93014489 \end{bmatrix}$$

### Sum of elements of $a_1$
$$\text{Sum}(a_1) = 1.5350184 + 1.98250233 + 1.93014489 = 5.4477$$

*   **Answer**: **Option B (5.44)**

---

## 3. Question 3: Post-activation Vector $h_1$

### Mathematical Formula
The post-activation vector is the element-wise sigmoid activation of the pre-activation vector:
$$h_1 = \sigma(a_1) = \frac{1}{1 + e^{-a_1}}$$

### Calculation
Using the elements of $a_1$:
1.  $h_{1,1} = \frac{1}{1 + e^{-1.5350184}} = 0.822739$
2.  $h_{1,2} = \frac{1}{1 + e^{-1.98250233}} = 0.878948$
3.  $h_{1,3} = \frac{1}{1 + e^{-1.93014489}} = 0.873265$

Thus:
$$h_1 = \begin{bmatrix} 0.82273938 \\ 0.87894766 \\ 0.87326546 \end{bmatrix}$$

### Sum of elements of $h_1$
$$\text{Sum}(h_1) = 0.82273938 + 0.87894766 + 0.87326546 = 2.57495$$

*   **Answer**: **Option A (2.57)**

---

## 4. Question 4: Forward Pass and Loss

To calculate the loss, we first complete the forward propagation.

### Hidden Layer 2 Calculations
1.  **Pre-activations**: $a_2 = W_2 h_1 + b_2$
    Using $W_2$ and $b_2$ from the parameters dump:
    $$a_2 = \begin{bmatrix} 2.10091008 \\ 1.76019958 \\ 2.59905711 \end{bmatrix} \Rightarrow \text{Sum}(a_2) = 6.46$$

2.  **Post-activations**: $h_2 = \sigma(a_2)$
    $$h_2 = \begin{bmatrix} 0.89099238 \\ 0.85324546 \\ 0.9308477 \end{bmatrix} \Rightarrow \text{Sum}(h_2) = 2.63$$

### Output Layer Calculations
1.  **Pre-activations**: $a_3 = W_3 h_2 + b_3$
    $$a_3 = \begin{bmatrix} 1.34960588 \\ 1.70564619 \\ 1.81966891 \end{bmatrix} \Rightarrow \text{Sum}(a_3) = 4.87$$

2.  **Softmax predictions**: $\hat{y} = \text{softmax}(a_3)$
    $$\hat{y}_i = \frac{e^{a_{3,i}}}{e^{a_{3,1}} + e^{a_{3,2}} + e^{a_{3,3}}}$$
    
    *   $e^{a_{3,1}} = e^{1.34960588} = 3.8559$
    *   $e^{a_{3,2}} = e^{1.70564619} = 5.5050$
    *   $e^{a_{3,3}} = e^{1.81966891} = 6.1700$
    *   Sum of exponents = $3.8559 + 5.5050 + 6.1700 = 15.5309$
    
    Dividing each exponent by the sum:
    $$\hat{y} = \begin{bmatrix} 0.23691422 \\ 0.33838847 \\ 0.42469732 \end{bmatrix}$$

### Loss Calculation (Cross-Entropy)
Since the label is one-hot encoded as $y = [0, 0, 1]^T$, the loss simplifies to:
$$L = -\ln(\hat{y}_3) = -\ln(0.42469732) = 0.856379$$

*   **Answer**: **`0.856379`** (rounds to **0.86**)

---

## 5. Question 5: Gradient with respect to Output Pre-activations ($\nabla_{a_3} L$)

### Mathematical Derivation
We want to find the partial derivatives of the cross-entropy loss $L$ with respect to the output pre-activations $a_{3,i}$.
For a true label class $k$ where $y_k = 1$, it can be proven that the gradient of Softmax + Cross Entropy simplifies beautifully:
$$\frac{\partial L}{\partial a_{3,i}} = \hat{y}_i - y_i$$

In vector form:
$$\nabla_{a_3} L = \hat{y} - y$$

### Calculation
Given:
$$\hat{y} = \begin{bmatrix} 0.23691422 \\ 0.33838847 \\ 0.42469732 \end{bmatrix}, \quad y = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

$$\nabla_{a_3} L = \begin{bmatrix} 0.23691422 \\ 0.33838847 \\ 0.42469732 \end{bmatrix} - \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.23691422 \\ 0.33838847 \\ -0.57530268 \end{bmatrix}$$

*   **Answer**: **Option E** ($[ 0.23, 0.33, -0.57 ]^T$)

---

## 6. Question 6: Gradient with respect to Bias ($\nabla_{b_2} L$)

### Mathematical Derivation (Backpropagating through layer 2)
We want to find the gradient of the loss with respect to the bias of the second hidden layer, $b_2$.
Using the chain rule:
$$\nabla_{b_2} L = \nabla_{a_2} L \odot \frac{\partial a_2}{\partial b_2}$$

Since $a_2 = W_2 h_1 + b_2$, we have $\frac{\partial a_2}{\partial b_2} = I$ (the identity matrix). Hence:
$$\nabla_{b_2} L = \nabla_{a_2} L$$

Let's find $\nabla_{a_2} L$:
$$\nabla_{a_2} L = \nabla_{h_2} L \odot \sigma'(a_2)$$

Where:
1.  **Gradient backpropagated from layer 3**:
    Since $a_3 = W_3 h_2 + b_3$, the derivative with respect to $h_2$ is:
    $$\nabla_{h_2} L = W_3^T \nabla_{a_3} L$$
2.  **Sigmoid Derivative**:
    $$\sigma'(a_2) = h_2 \odot (1 - h_2)$$

Combining these:
$$\nabla_{b_2} L = (W_3^T (\hat{y} - y)) \odot h_2 \odot (1 - h_2)$$

where $\odot$ denotes element-wise multiplication.

### Calculation
Using the computed values:
1.  $W_3^T (\hat{y} - y)$:
    $$= \begin{bmatrix} 0.11827 & 0.94467 & 0.26456 \\ 0.63992 & 0.52185 & 0.77423 \\ 0.14335 & 0.41466 & 0.45615 \end{bmatrix} \begin{bmatrix} 0.23691 \\ 0.33839 \\ -0.57530 \end{bmatrix} = \begin{bmatrix} 0.19535 \\ -0.11718 \\ -0.08846 \end{bmatrix}$$

2.  $h_2 \odot (1 - h_2)$:
    $$= \begin{bmatrix} 0.89099 \times 0.10901 \\ 0.85325 \times 0.14675 \\ 0.93085 \times 0.06915 \end{bmatrix} = \begin{bmatrix} 0.09712 \\ 0.12522 \\ 0.06437 \end{bmatrix}$$

3.  $\nabla_{b_2} L$:
    $$= \begin{bmatrix} 0.19535 \\ -0.11718 \\ -0.08846 \end{bmatrix} \odot \begin{bmatrix} 0.09712 \\ 0.12522 \\ 0.06437 \end{bmatrix} = \begin{bmatrix} 0.01838 \\ -0.01998 \\ -0.00384 \end{bmatrix}$$

*   **Answer**: **Option A** ($[ 0.018, -0.019, -0.003 ]^T$)

---

## 7. Question 7: Gradient Updates and New Loss ($\eta = 1$)

We calculate the gradients for all weights and biases in the network and update them using gradient descent:
$$\theta_{\text{new}} = \theta - \eta \nabla_{\theta} L$$

### Gradient Formulas for All Parameters
*   **Layer 3**:
    $$\nabla_{W_3} L = \nabla_{a_3} L h_2^T$$
    $$\nabla_{b_3} L = \nabla_{a_3} L$$
*   **Layer 2**:
    $$\nabla_{W_2} L = \nabla_{a_2} L h_1^T$$
    $$\nabla_{b_2} L = \nabla_{a_2} L$$
*   **Layer 1**:
    $$\nabla_{a_1} L = (W_2^T \nabla_{a_2} L) \odot h_1 \odot (1 - h_1)$$
    $$\nabla_{W_1} L = \nabla_{a_1} L x^T$$
    $$\nabla_{b_1} L = \nabla_{a_1} L$$

### Parameter Updates ($\eta = 1$)
Subtract the computed gradient from each corresponding parameter matrix:
*   $W_{\text{layer, new}} = W_{\text{layer}} - \nabla_{W_{\text{layer}}} L$
*   $b_{\text{layer, new}} = b_{\text{layer}} - \nabla_{b_{\text{layer}}} L$

### New Forward Pass and Loss
Propagating the input $x = [1, 0, 1]^T$ through the network with updated parameters yields:
1.  **Layer 1 new**: 
    $$a_{1,\text{new}} = \begin{bmatrix} 1.62479 \\ 2.08027 \\ 2.01529 \end{bmatrix} \Rightarrow h_{1,\text{new}} = \begin{bmatrix} 0.83546 \\ 0.88898 \\ 0.88237 \end{bmatrix}$$

2.  **Layer 2 new**: 
    $$a_{2,\text{new}} = \begin{bmatrix} 2.11589 \\ 1.74542 \\ 2.59560 \end{bmatrix} \Rightarrow h_{2,\text{new}} = \begin{bmatrix} 0.89246 \\ 0.85137 \\ 0.93060 \end{bmatrix}$$

3.  **Layer 3 new**: 
    $$a_{3,\text{new}} = \begin{bmatrix} 1.13854 \\ 1.41743 \\ 4.09204 \end{bmatrix} \Rightarrow \hat{y}_{\text{new}} = \begin{bmatrix} 0.048383 \\ 0.063943 \\ 0.897674 \end{bmatrix}$$

### New Cross-Entropy Loss
$$L_{\text{new}} = -\ln(\hat{y}_{3,\text{new}}) = -\ln(0.897674) = 0.072533$$

*   **Answer**: **`0.072533`** (rounds to **0.07**)
