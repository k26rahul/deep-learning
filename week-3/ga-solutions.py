from pathlib import Path
import numpy as np


def load_parameters():
    """Load neural network parameters from the npz file."""
    parameters_path = Path(__file__).parent / "parameters.npz"
    return np.load(parameters_path)


if __name__ == "__main__":
    parameters = load_parameters()
    
    # Input vector x
    x = np.array([[1], [0], [1]])
    
    # Question 2: Calculate the pre-activation vector a1 = W1 * x + b1
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    a1 = np.dot(W1, x) + b1
    
    print("Question 2:")
    print(f"a1:\n{a1}")
    print(f"Sum of elements of a1: {np.sum(a1):.2f}")

    # Question 3: Calculate the post-activation vector h1 = sigmoid(a1)
    h1 = 1 / (1 + np.exp(-a1))
    print("\nQuestion 3:")
    print(f"h1:\n{h1}")
    print(f"Sum of elements of h1: {np.sum(h1):.2f}")

    # Question 4: Complete forward pass and calculate loss
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # Second hidden layer
    a2 = np.dot(W2, h1) + b2
    h2 = 1 / (1 + np.exp(-a2))
    
    # Output layer
    a3 = np.dot(W3, h2) + b3
    y_hat = np.exp(a3) / np.sum(np.exp(a3), axis=0)
    
    # Loss value (y label is one-hot vector with 1 at index 2)
    y = np.array([[0], [0], [1]])
    loss = -np.log(y_hat[2, 0])
    
    # Question 5: Gradient of loss with respect to a3: grad_a3 = y_hat - y
    grad_a3 = y_hat - y
    
    print("\nQuestion 4:")
    print(f"Sum of elements of a2: {np.sum(a2):.2f} (Expected: 6.4)")
    print(f"Sum of elements of h2: {np.sum(h2):.2f} (Expected: 2.63)")
    print(f"Sum of elements of a3: {np.sum(a3):.2f} (Expected: 4.87)")
    print(f"y_hat:\n{y_hat}")
    print(f"Loss value: {loss:.6f}")
    
    print("\nQuestion 5:")
    print(f"grad_a3:\n{grad_a3}")

    # Question 6: Gradient of loss with respect to b2: grad_b2
    # grad_b2 = (W3.T * grad_a3) * h2 * (1 - h2)
    grad_h2 = np.dot(W3.T, grad_a3)
    grad_b2 = grad_h2 * h2 * (1 - h2)
    
    print("\nQuestion 6:")
    print(f"grad_b2:\n{grad_b2}")

    # Question 7: Complete backpropagation, update parameters with eta=1,
    # and calculate the new loss.
    
    # Gradients for Layer 3
    grad_W3 = np.dot(grad_a3, h2.T)
    grad_b3 = grad_a3
    
    # Gradients for Layer 2
    grad_W2 = np.dot(grad_b2, h1.T)
    # grad_b2 is already calculated above
    
    # Gradients for Layer 1
    grad_h1 = np.dot(W2.T, grad_b2)
    grad_a1 = grad_h1 * h1 * (1 - h1)
    grad_W1 = np.dot(grad_a1, x.T)
    grad_b1 = grad_a1
    
    # Parameter updates (eta = 1)
    eta = 1
    W1_new = W1 - eta * grad_W1
    b1_new = b1 - eta * grad_b1
    W2_new = W2 - eta * grad_W2
    b2_new = b2 - eta * grad_b2
    W3_new = W3 - eta * grad_W3
    b3_new = b3 - eta * grad_b3
    
    # New forward pass
    a1_new = np.dot(W1_new, x) + b1_new
    h1_new = 1 / (1 + np.exp(-a1_new))
    
    a2_new = np.dot(W2_new, h1_new) + b2_new
    h2_new = 1 / (1 + np.exp(-a2_new))
    
    a3_new = np.dot(W3_new, h2_new) + b3_new
    y_hat_new = np.exp(a3_new) / np.sum(np.exp(a3_new), axis=0)
    
    # New loss
    loss_new = -np.log(y_hat_new[2, 0])
    
    print("\nQuestion 7:")
    print(f"New Loss value (eta = 1): {loss_new:.6f}")
