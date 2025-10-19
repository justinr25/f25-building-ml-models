### NEURAL NETWORK CLASS ###
# Just for demo. We'll write a nice neat description later.
import autodiff
import random
import math

class ANN_arbor():
    """Neural network class, powered by in-python automatic differentiation.
    
        Arguments:
            n_in (int): number of input nodes, corresponding to the number of features in a dataset. 
            n_hl (int): number of hidden nodes. Increasing this gives the ANN more flexibility, but increases overfitting and runtime.
            n_ot (int): number of output nodes.
            n_bt (int): number of bias terms. 
            hl_af (expression): hidden layer activation function.
            ol_af (expression): output layer activation function. 
                                Regression: use linear activation
                                Classification: Sigma RAHHH
            lf (function): loss function (should just be MSE for now)
    """
    def __init__(self, n_in, n_ot, n_hl, n_bt, hl_af, ol_af, lf):
        self.n_in = n_in
        self.n_hl = n_hl
        self.n_ot = n_ot
        self.n_bt = n_bt
        self.hl_af = hl_af
        self.ol_af = ol_af
        self.lf = lf
        
        # inputs, weights, bias initialization
        # weights naming system: w_layer_from_to
        # bias naming system: b_layer_node
        self.weights = {}
        self.biases = {}
        self.inputs = {}
        self.outputs = {}
        
        # create input variables
        for i in range(n_in):
            self.inputs[f'x_{i}'] = autodiff.expression(f'x_{i}')
        
        # initialize weights with small random values
        # Input to hidden layer weights
        for i in range(n_in):
            for h in range(n_hl):
                weight_name = f'w_0_{i}_{h}'
                self.weights[weight_name] = autodiff.expression(weight_name)
        
        # Hidden to output layer weights
        for h in range(n_hl):
            for o in range(n_ot):
                weight_name = f'w_1_{h}_{o}'
                self.weights[weight_name] = autodiff.expression(weight_name)
        
        # initialize biases
        for h in range(n_hl):
            bias_name = f'b_0_{h}'
            self.biases[bias_name] = autodiff.expression(bias_name)
        
        for o in range(n_ot):
            bias_name = f'b_1_{o}'
            self.biases[bias_name] = autodiff.expression(bias_name)
        
    def feedforward(self, X):
        """Feedforward function. 
        
            Arguments:
                X (array-like): input data, shape (n_samples, n_features)
                
            Returns:
                output_expressions (list): list of autodiff expressions representing outputs
        """
        
        # Input-hidden
        hidden_outputs = []
        for h in range(self.n_hl):
            # Calculate weighted sum for this hidden node
            weighted_sum = self.biases[f'b_0_{h}']
            for i in range(self.n_in):
                weighted_sum = weighted_sum + self.weights[f'w_0_{i}_{h}'] * self.inputs[f'x_{i}']
            
            # Apply activation function
            if self.hl_af == 'sigmoid':
                activated = autodiff.sigmoid(weighted_sum)
            elif self.hl_af == 'relu':
                activated = autodiff.relu(weighted_sum)
            elif self.hl_af == 'tanh':
                activated = autodiff.hyperbolic_tan(weighted_sum)
            elif self.hl_af == 'linear':
                activated = autodiff.linear(weighted_sum)
            else:
                activated = weighted_sum  # default to linear (easy)
            
            hidden_outputs.append(activated)
        
        # Hidden-output, including bias terms
        output_expressions = []
        for o in range(self.n_ot):
            weighted_sum = self.biases[f'b_1_{o}']
            for h in range(self.n_hl):
                weighted_sum = weighted_sum + self.weights[f'w_1_{h}_{o}'] * hidden_outputs[h]
            
            # Apply activation function
            if self.ol_af == 'sigmoid':
                activated = autodiff.sigmoid(weighted_sum)
            elif self.ol_af == 'relu':
                activated = autodiff.relu(weighted_sum)
            elif self.ol_af == 'tanh':
                activated = autodiff.hyperbolic_tan(weighted_sum)
            elif self.ol_af == 'linear':
                activated = autodiff.linear(weighted_sum)
            else:
                activated = weighted_sum  # default to linear
            
            output_expressions.append(activated)
        
        return output_expressions

    
    def compute_loss(self, y_true_values, output_expressions, values_dict):
        """Compute loss using autodiff expressions.
        
            Arguments:
                y_true_values (list): true target values
                output_expressions (list): autodiff expressions for network outputs
                values_dict (dict): current values of all variables
                
            Returns:
                loss_expression: autodiff expression representing the loss
        """
        if self.lf == 'mse':
            # Mean Squared Error
            loss = autodiff.constant(0)
            for i in range(len(y_true_values)):
                diff = output_expressions[i] - autodiff.constant(y_true_values[i])
                squared_diff = diff * diff
                loss = loss + squared_diff
            # Average over outputs
            loss = loss / autodiff.constant(len(y_true_values))
            return loss
        else:
            raise ValueError(f"Unsupported loss function: {self.lf}")
    
    def backpropagate(self, y_true, output_expressions, values_dict):
        """Backpropagation function using automatic differentiation.
        
            Arguments:
                y_true (list): true labels
                output_expressions (list): autodiff expressions for network outputs  
                values_dict (dict): current values of all variables (inputs, weights, biases)
                
            Returns:
                gradients (dict): gradients of loss with respect to weights and biases
        """
        
        # Compute loss expression
        loss_expr = self.compute_loss(y_true, output_expressions, values_dict)
        
        # Compute gradients using automatic differentiation
        gradients = {}
        
        # Gradients with respect to weights
        for weight_name in self.weights:
            gradients[weight_name] = loss_expr.diff(values_dict, weight_name)
        
        # Gradients with respect to biases  
        for bias_name in self.biases:
            gradients[bias_name] = loss_expr.diff(values_dict, bias_name)
        
        return gradients
    
    def initialize_weights(self, seed=None):
        """Initialize weights and biases with small random values.
        
            Arguments:
                seed (int): random seed for reproducibility
        """
        if seed:
            random.seed(seed)
        
        values = {}
        
        # Initialize weights with small random values
        for weight_name in self.weights:
            values[weight_name] = random.uniform(-0.5, 0.5)
        
        # Initialize biases with small random values
        for bias_name in self.biases:
            values[bias_name] = random.uniform(-0.1, 0.1)
        
        return values
    
    def predict(self, X, values_dict):
        """Make predictions using current weights and biases.
        
            Arguments:
                X (list): input data, shape (n_features,)
                values_dict (dict): current values of all variables
                
            Returns:
                predictions (list): predicted outputs
        """
        # Set input values
        for i, val in enumerate(X):
            values_dict[f'x_{i}'] = val
        
        # Get output expressions
        output_expressions = self.feedforward(X)
        
        # Evaluate output expressions
        predictions = []
        for expr in output_expressions:
            predictions.append(expr.eval(values_dict))
        
        return predictions
    
    def train_step(self, X, y_true, values_dict, learning_rate=0.01):
        """Perform one training step (forward + backward + update).
        
            Arguments:
                X (list): input data, shape (n_features,)
                y_true (list): true target values
                values_dict (dict): current values of all variables
                learning_rate (float): learning rate for gradient descent
                
            Returns:
                loss_value (float): current loss value
                updated_values_dict (dict): updated parameter values
        """
        # Set input values
        for i, val in enumerate(X):
            values_dict[f'x_{i}'] = val
        
        # Forward pass
        output_expressions = self.feedforward(X)
        
        # Compute loss
        loss_expr = self.compute_loss(y_true, output_expressions, values_dict)
        loss_value = loss_expr.eval(values_dict)
        
        # Backward pass
        gradients = self.backpropagate(y_true, output_expressions, values_dict)
        
        # Update parameters using gradient descent
        for param_name in gradients:
            values_dict[param_name] -= learning_rate * gradients[param_name]
        
        return loss_value, values_dict
    
    def train(self, training_data, epochs=1000, learning_rate=0.1, verbose=True):
        """Train the neural network on provided data.
        
            Arguments:
                training_data (list): list of (input, target) tuples
                epochs (int): number of training epochs
                learning_rate (float): learning rate for gradient descent
                verbose (bool): whether to print training progress
                
            Returns:
                values_dict (dict): trained parameter values
                loss_history (list): history of loss values
        """
        values_dict = self.initialize_weights()
        loss_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            for X, y_true in training_data:
                loss, values_dict = self.train_step(X, y_true, values_dict, learning_rate)
                total_loss += loss
            
            avg_loss = total_loss / len(training_data)
            loss_history.append(avg_loss)
            
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
        
        return values_dict, loss_history
        

def main():
    print("XOR Problem (Classification)")
    
    nn_xor = ANN_arbor(
        n_in=2, n_ot=1, n_hl=4, n_bt=1,
        hl_af='sigmoid', ol_af='linear', lf='mse'
    )
    
    xor_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]
    
    # Train the network
    values_dict, loss_history = nn_xor.train(xor_data, epochs=1000, learning_rate=0.5)
    
    # Test predictions
    print("\nXOR Results:")
    for X, y_true in xor_data:
        predictions = nn_xor.predict(X, values_dict)
        error = abs(predictions[0] - y_true[0])
        print(f"Input: {X}, True: {y_true[0]:.1f}, Predicted: {predictions[0]:.4f}, Error: {error:.4f}")
    
    # # Example 2: Simple Regression
    # print("\n\n2. Simple Function Approximation (Regression)")
    # print("-" * 45)
    
    # nn_reg = ANN_arbor(
    #     n_in=1, n_ot=1, n_hl=5, n_bt=1,
    #     hl_af='tanh', ol_af='linear', lf='mse'
    # )
    
    # # Generate data for y = x^2
    # regression_data = []
    # for x in [-2, -1, -0.5, 0, 0.5, 1, 2]:
    #     regression_data.append(([x], [x * x]))
    
    # # Train the network
    # values_dict_reg, _ = nn_reg.train(regression_data, epochs=2000, learning_rate=0.01, verbose=False)
    
    # # Test predictions
    # print("Function Approximation Results (y = x^2):")
    # test_points = [-1.5, -0.25, 0.75, 1.5]
    # for x in test_points:
    #     true_y = x * x
    #     pred_y = nn_reg.predict([x], values_dict_reg)[0]
    #     error = abs(pred_y - true_y)
    #     print(f"x: {x:5.2f}, True: {true_y:5.2f}, Predicted: {pred_y:5.2f}, Error: {error:.4f}")
    
    # print(f"\nFinal training loss: {loss_history[-1]:.6f}")
    
if __name__ == "__main__":
    main()
    