import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

def relu(Z):
    """ReLU activation function: max(0, Z)"""
    return np.maximum(0,Z)

def sigmoid(z):
    """Sigmoid activation function: 1/(1 + e^(-Z))"""
    return 1/(1+np.exp(-z))

def softmax(Z):
   """Softmax activation function for multi-class classification"""
   # Subtract max for numerical stability (prevents overflow)
   exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
   return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

class NeuralNetwork:
   def __init__(self, layer_dims, activations):
       """
       Initialize a neural network with specified layer dimensions and activations
      
       Parameters:
       - layer_dims: List of integers representing the number of neurons in each layer
                    (including input and output layers)
       - activations: List of activation functions for each layer (excluding input layer)
       """
       self.L = len(layer_dims) - 1  # Number of layers (excluding input layer)
       self.layer_dims = layer_dims
       self.activations = activations
       self.parameters = {}
      
       # Initialize parameters (weights and biases)
       self.initialize_parameters()
  
   def initialize_parameters(self):
       """Initialize weights and biases with small random values"""
       for l in range(1, self.L + 1):
           # He initialization for weights - helps with training deep networks
           self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])
           self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
  
   def forward_propagation(self, X):
       """
       Perform forward propagation through the network
      
       Parameters:
       - X: Input data (n_features, batch_size)
      
       Returns:
       - AL: Output of the network
       - caches: Dictionary containing all activations and pre-activations
       """
       caches = {}
       A = X  # Input layer activation
       caches['A0'] = X
      
       # Process through L-1 layers (excluding output layer)
       for l in range(1, self.L):
           A_prev = A
          
           # Get weights and biases for current layer
           W = self.parameters[f'W{l}']
           b = self.parameters[f'b{l}']
          
           # Forward propagation for current layer
           Z = np.dot(W, A_prev) + b
          
           # Apply activation function
           activation_function = self.activations[l-1]
           if activation_function == "relu":
               A = relu(Z)
           elif activation_function == "sigmoid":
               A = sigmoid(Z)
          
           # Store values for backpropagation
           caches[f'Z{l}'] = Z
           caches[f'A{l}'] = A
      
       # Output layer
       W = self.parameters[f'W{self.L}']
       b = self.parameters[f'b{self.L}']
       Z = np.dot(W, A) + b
      
       # Apply output activation function
       activation_function = self.activations[self.L-1]
       if activation_function == "sigmoid":
           AL = sigmoid(Z)
       elif activation_function == "softmax":
           AL = softmax(Z)
       elif activation_function == "linear":
           AL = Z  # No activation for regression
      
       # Store output layer values
       caches[f'Z{self.L}'] = Z
       caches[f'A{self.L}'] = AL
      
       return AL, caches

if __name__=="__main__":
    # Create a sample network
    # Input layer: 3 features
    # Hidden layer 1: 4 neurons with ReLU activation
    # Output layer: 2 neurons with sigmoid activation (binary classification)
    layer_dims = [3, 4, 2]
    activations = ["relu", "sigmoid"]
    nn = NeuralNetwork(layer_dims, activations)
    # Create sample input data - 3 features for 5 examples
    X = np.random.randn(3, 5)
    # Perform forward propagation
    output, caches = nn.forward_propagation(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values:\n{output}")


