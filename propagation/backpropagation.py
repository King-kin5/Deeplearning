import numpy as np

np.random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layer 1: Input to Hidden1
        self.weights_input_hidden1 = np.random.randn(
            self.input_size, self.hidden_size)
        self.bias_hidden1 = np.zeros((1, self.hidden_size))
        
        # Layer 2: Hidden1 to Hidden2
        self.weights_hidden1_hidden2 = np.random.randn(
            self.hidden_size, self.hidden_size)
        self.bias_hidden2 = np.zeros((1, self.hidden_size))
        
        # Layer 3: Hidden2 to Hidden3
        self.weights_hidden2_hidden3 = np.random.randn(
            self.hidden_size, self.hidden_size)
        self.bias_hidden3 = np.zeros((1, self.hidden_size))
        
        # Layer 4: Hidden3 to Output
        self.weights_hidden3_output = np.random.randn(
            self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self,X):
        # Layer 1: Input to Hidden1
        self.hidden1_activation = np.dot(
          X, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden1_output = self.sigmoid(self.hidden1_activation)
        
        # Layer 2: Hidden1 to Hidden2
        self.hidden2_activation = np.dot(
          self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden2_output = self.sigmoid(self.hidden2_activation)
        
        # Layer 3: Hidden2 to Hidden3
        self.hidden3_activation = np.dot(
          self.hidden2_output, self.weights_hidden2_hidden3) + self.bias_hidden3
        self.hidden3_output = self.sigmoid(self.hidden3_activation)
        
        # Layer 4: Hidden3 to Output
        self.output_activation = np.dot(
          self.hidden3_output, self.weights_hidden3_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output
    
    def backward(self,X,y,learning_rate):
      # Output layer error and delta
      output_error = y - self.predicted_output
      output_delta = output_error * self.sigmoid_derivative(self.predicted_output)
      
      # Hidden3 layer error and delta
      hidden3_error = np.dot(output_delta, self.weights_hidden3_output.T)
      hidden3_delta = hidden3_error * self.sigmoid_derivative(self.hidden3_output)
      
      # Hidden2 layer error and delta
      hidden2_error = np.dot(hidden3_delta, self.weights_hidden2_hidden3.T)
      hidden2_delta = hidden2_error * self.sigmoid_derivative(self.hidden2_output)
      
      # Hidden1 layer error and delta
      hidden1_error = np.dot(hidden2_delta, self.weights_hidden1_hidden2.T)
      hidden1_delta = hidden1_error * self.sigmoid_derivative(self.hidden1_output)

      # Update weights and biases: Hidden3 to Output
      self.weights_hidden3_output += np.dot(self.hidden3_output.T, output_delta) * learning_rate
      self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
      
      # Update weights and biases: Hidden2 to Hidden3
      self.weights_hidden2_hidden3 += np.dot(self.hidden2_output.T, hidden3_delta) * learning_rate
      self.bias_hidden3 += np.sum(hidden3_delta, axis=0, keepdims=True) * learning_rate
      
      # Update weights and biases: Hidden1 to Hidden2
      self.weights_hidden1_hidden2 += np.dot(self.hidden1_output.T, hidden2_delta) * learning_rate
      self.bias_hidden2 += np.sum(hidden2_delta, axis=0, keepdims=True) * learning_rate
      
      # Update weights and biases: Input to Hidden1
      self.weights_input_hidden1 += np.dot(X.T, hidden1_delta) * learning_rate
      self.bias_hidden1 += np.sum(hidden1_delta, axis=0, keepdims=True) * learning_rate

    def train(self,X,y,epochs,learning_rate):
        for epoch in range(epochs):
            #computes the output for the current inputs
            output = self.feedforward(X)

            # updates weights and biases using Back Propagation
            self.backward(X, y, learning_rate)

            # calculates the mean squared error (MSE) loss
            if epoch % 4000 == 0:
               loss = np.mean(np.square(y - output))
               print(f"Epoch {epoch}, Loss:{loss}")

if __name__=="__main__":
    num_samples = 1000000
    input_size = 10  # Number of features
    output_size = 1  # Single output
    
    X = np.random.randn(num_samples, input_size)
    y = np.random.randint(0, 2, (num_samples, output_size))  
    
    # Initialize and train the neural network
    nn = NeuralNetwork(input_size=input_size, hidden_size=64, output_size=output_size)
    nn.train(X, y, epochs=10000, learning_rate=0.1)
