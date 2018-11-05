import numpy as np
import os

# Implementation of the MLP class
class MLP:

    def __init__(self):
        '''
        Initialize the MLP
        
        Attributes:       
        - Parameters: dictionary containing weights and biases
        - NumberLayers: number of layers of the MLP
        '''
        
        self.parameters = {}
        self.numberLayers = 0
	
    def initialize_parameters_layer(self, numberNeuronsPreviousLayer, numberNeuronsCurrentLayer, numberLayer):
        '''
        Initialize the weights and biases of a layer
        
        Arguments:
        - numberNeuronsPreviousLayer
        - numberNeuronsCurrentLayer
        - numberLayer: number of the current layer
        '''

        # Weights initialization -> 2015 by He et al (similar to Xavier initialization)
        W_layer = np.random.randn(numberNeuronsCurrentLayer, numberNeuronsPreviousLayer)*np.sqrt(2/numberNeuronsPreviousLayer)
		
        # Biases initialization -> array of zeros
        b_layer = np.zeros(shape=(numberNeuronsCurrentLayer, 1))
        
        # Add the weights and biases in the parameters dict
        self.parameters['W' + str(numberLayer)] = W_layer
        self.parameters['b' + str(numberLayer)] = b_layer
	
    def create_layer(self, numberNeuronsLayer):
        '''
        Create a layer in the MLP
        
        Argument:
        - numberNeuronsLayer: number of neurons of the created layer
        '''

        # Hidden Layer or Output Layer
        if self.numberLayers > 0:
            self.initialize_parameters_layer(self.numberNeuronsPreviousLayer, numberNeuronsLayer, self.numberLayers)
			
        self.numberNeuronsPreviousLayer = numberNeuronsLayer
        self.numberLayers += 1
		
    def forward_propagation(self, X):
        """
        Implement forward propagation

        Argument:
        - X: input data

        Returns:
        - AL: activation of the output layer
        - cache: dict containing activation potential and activation of each layer to compute the gradients in the back-propagation stage
        """

        # Dict cache
        cache = {}

        for layer in range(1,self.numberLayers):
            # Recover parameters
            W_layer = self.parameters['W' + str(layer)]
            b_layer = self.parameters['b' + str(layer)]
			
			      # Activation Potential
            if layer == 1: # Compute linear activation from the input layer to the first hidden layer
                Z = np.dot(W_layer, X.T) + b_layer
            else: # Compute linear activation from some hidden layer to a hidden layer or the output layer
                Z = np.dot(W_layer, cache['A' + str(layer - 1)]) + b_layer
			
            # Activation of the layer -> execute activation Function
            A = self.relu(Z)
			
			      # Stores values in the cache
            cache['Z' + str(layer)] = Z
            cache['A' + str(layer)] = A
         
        # Activation of the output layer
        AL = cache['A' + str(layer)]
        
        return AL, cache
		
    def back_propagation(self, cache, X, Y):
        """
        Implement the backward propagation

        Arguments:
        - cache: dict containing activation potential and activation of each layer
        - AL: output of the forward propagation (activation of the output layer)
        - Y: true "label" vector
        
        Returns:
        - deltas: dict containing deltas of each layer (weights and biases)
        """
        
        # Dict deltas
        deltas = {}
		
		    # Number of examples
        m = X.shape[1]

		
        # Backward propagation: compute deltas for the neurons of all layers #	
       
		
		    ## Compute error of the output layer (value obtained - desired value)
        dZ_output_layer = cache['A' + str(self.numberLayers - 1)] - Y.T
		
		    ## Delta for the output layer weights: error * derivative_function(output value) * input value(output of the previous layer)
        dW_output_layer = (1 / m) * -1 * np.dot(np.multiply(dZ_output_layer, self.relu_derivative(cache['A' + str(self.numberLayers - 1)])), cache['A' + str(self.numberLayers - 2)].T)
		
		    ## Delta for the output layer bias
        db_output_layer = (1 / m) * -1 * np.sum(dZ_output_layer, axis=1, keepdims=True)
		
        dZ_previous_layer = dZ_output_layer
		
        deltas['dW' + str(self.numberLayers - 1)] = dW_output_layer
        deltas['db' + str(self.numberLayers - 1)] = db_output_layer
		
        for layer in reversed(range(1,self.numberLayers - 1)):	

			      ## Gradient of the hidden layer
            dZ_hidden_layer = np.multiply(np.dot(self.parameters['W' + str(layer + 1)].T, dZ_previous_layer), self.relu_derivative(cache['A' + str(layer)]))
		
		        ## Delta for the hidden layer weights
            if layer == 1:
                dW_hidden_layer = (1 / m) * -1 * np.dot(dZ_hidden_layer, X)
            else:
                dW_hidden_layer = (1 / m) * -1 * np.dot(dZ_hidden_layer, cache['A' + str(layer-1)].T)
		
		        ## Delta for the hidden layer bias
            db_hidden_layer = (1 / m) * -1 * np.sum(dZ_hidden_layer, axis=1, keepdims=True)
			
            dZ_previous_layer = dZ_hidden_layer

            deltas['dW' + str(layer)] = dW_hidden_layer
            deltas['db' + str(layer)] = db_hidden_layer
		
		    #

        return deltas
		
    def update_parameters(self, deltas, learning_rate=0.1, momentum=1):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of L_model_backward

        Returns:
        parameters -- python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...
        """
        
        # Dict of the new parameters (updated weights and biases)
        new_parameters = {}
		
        # Update each parameter (weights and biases)
        for layer in range(1,self.numberLayers):
            new_parameters['W' + str(layer)] = (self.parameters['W' + str(layer)] * 1) + (learning_rate * deltas['dW' + str(layer)])
            new_parameters['b' + str(layer)] = (self.parameters['b' + str(layer)] * 1) + (learning_rate * deltas['db' + str(layer)])
			
        self.parameters = new_parameters
		
    def cost_function_mse(self, AL, Y):
        """
        Implement the mean squared error (mse) cost function

        Arguments:
        - AL: output of the forward propagation (label predictions)
        - Y: true "label" vector

        Returns:
        - mse: error
        """
        
        Y = Y.T
        
        # Number of examples
        m = Y.shape[1]        
        
        # Compute mean squared error (mse)
        mse = np.divide(np.sum(np.power(np.subtract(Y, AL), 2)), m)
    
        # Assert that is a float value
        assert(isinstance(mse, float))
        
        return mse
			
    def training_mlp_any_layers_with_mse(self, X, Y, learning_rate=0.1, epsilon=1e-15):
        """
        Implement the training stage of the MLP based on MSE

        Arguments:
        - X: input data
        - Y: true "label" vector
        - learning_rate: used to measure update parameters
        - epsilon: precision to stop training

        """
        
        current_mse_cost = 0
        
        # Loop
        while True:
		
            # MSE of the last epoch
            previous_mse_cost = current_mse_cost
             
            # Forward propagation
            A2, cache = self.forward_propagation(X)
			
            # Back-propagation
            deltas = self.back_propagation(cache, X, Y)
			
            # Compute MSE of this epoch
            current_mse_cost = self.cost_function_mse(A2, Y)
     
            # Update parameters
            self.update_parameters(deltas, learning_rate)
			
            print('MSE: ', current_mse_cost)
			
            # Stop when mse between two consecutive epochs is less or equal than an epsilon
            if np.abs(current_mse_cost - previous_mse_cost) <= epsilon:
                break
              
    def training_mlp_any_layers_with_number_iterations(self, X, Y, number_iterations=100000, learning_rate=0.1):
        """
        Implement the training stage of the MLP based on number of iterations

        Arguments:
        - X: input data
        - Y: true "label" vector
        - number_iterations: number of iterations used to train the MLP
        - learning_rate: used to measure update parameters
        - epsilon: precision to stop training

        """
        
        current_mse_cost = 0
        
        # Loop
        for i in range(number_iterations):
		
            # MSE of the last epoch
            previous_mse_cost = current_mse_cost
             
            # Forward propagation
            AL, cache = self.forward_propagation(X)
			
            # Back-propagation
            deltas = self.back_propagation(cache, X, Y)
			
            # Compute MSE of this epoch
            current_mse_cost = self.cost_function_mse(AL, Y)
     
            # Update parameters
            self.update_parameters(deltas, learning_rate)
			
            print('Iteration: ', i, ' - MSE: ', current_mse_cost)

        # Save model
        np.save(os.path.join("TrainedModels", "model_test"), self.parameters)
			
    def predict(self, X):
        """
        Implement the test stage of the MLP

        Arguments:
        - X: input data

        Returns:
        - predictions: label predictions of the mlp
        """
        
        # Compute the predictions
        A2, cache = self.forward_propagation(X)
        predictions = A2.T
        
        return predictions
      
    def compute_accuracy(self, X, Y):
        """
        Compute the accuracy of the MLP

        Arguments:
        - X: input data
        - Y: true "label" vector

        Returns:
        - accuracy
        """

        # Compute the predictions
        predictions = self.predict(X)
        
        new_predictions = []

        # Transforms each prediction array
        for p in predictions:
            new_predictions.append(self.round_max_value_to_1(p))

        new_predictions = np.array(new_predictions)

        # Compute correct predictions
        correct_predictions = 0

        for i in range(predictions.shape[0]):
            if (new_predictions[i] == Y[i]).all():
                correct_predictions += 1

        # Compute accuracy
        accuracy = (correct_predictions / predictions.shape[0]) * 100

        return accuracy
      
    def round_max_value_to_1(self, array):
      """
      Transforms the array in a vector of 0's and 1

      Arguments:
      - array: original labels vector

      Returns:
      - new_array: transformed labels vector
      """
      
      # Founds the max value index
      index = np.argmax(array)
      
      # Round array (example: [0, 0.90, 0.1] to [0, 1, 0], but [0, 0.45, 0.35] goes [0, 0, 0])
      new_array = np.round(array, decimals=0)
      
      # Max index is equal to 1
      new_array[index] = 1
      
      return new_array

		
    # Some activations functions and derivatives
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x): # x is already the sigmoid value
        return x * (1.0 - x)
		
    def relu(self,x):
        return np.maximum(x, 0, x)
		
    def relu_derivative(self,x):
        return np.greater(x, 0).astype(int)
		
    def load_parameters(self, parameters):
        self.parameters = parameters
		
		
#################################### MAIN ####################################
### SOME EXAMPLES ##

# OR EXAMPLE
def mlp_or():

    mlp = MLP()
    
    # Create layers
    mlp.create_layer(2) # Input layer
    mlp.create_layer(10) # 1st hidden layer
    mlp.create_layer(1) # output layer

    # Data
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[0],[1],[1],[1]])

    # Training the MLP
    #mlp.training_mlp_any_layers_with_mse(inputs, labels) # with mse
    mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 2000)
    
    # Test the MLP
    predictions = mlp.predict(inputs)
    
    print('NN parameters: ', mlp.parameters)
    print('Test: ', predictions)

# AND EXAMPLE
def mlp_and():

    mlp = MLP()

    # Create layers
    mlp.create_layer(2) # Input layer
    mlp.create_layer(10) # 1st hidden layer
    mlp.create_layer(1) # output layer

    # Data
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[0],[0],[0],[1]])

    # Training the MLP
    #mlp.training_mlp_any_layers_with_mse(inputs, labels) # with mse
    mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 2000)
    
    # Test the MLP
    predictions = mlp.predict(inputs)
    
    print('NN parameters: ', mlp.parameters)
    print('Test: ', predictions)


# XOR EXAMPLE	
def mlp_xor():

    mlp = MLP()
	
    # Create layers
    mlp.create_layer(2) # Input layer
    mlp.create_layer(20) # 1st hidden layer
    mlp.create_layer(1) # output layer

    # Data
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([[0],[1],[1],[0]])

    # Training the MLP
    mlp.training_mlp_any_layers_with_mse(inputs, labels) # with mse
    #mlp.training_mlp_any_layers_with_number_iterations(inputs, labels, 10000)
    
    # Test the MLP
    predictions = mlp.predict(inputs)
    
    print('NN parameters: ', mlp.parameters)
    print('Test: ', predictions)
	
#mlp_or()