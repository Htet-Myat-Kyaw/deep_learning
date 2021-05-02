import numpy as np

class Layer(object):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) 
        self.bias = np.zeros((1, output_size))
        
        self.input       = None
        self.weighted_sum= None
        self.output      = None
        
    def forward_prop(self, input_data):
        self.input       = input_data
        self.weighted_sum= np.dot(self.input, self.weights) + self.bias
        self.output      = sigmoid(self.weighted_sum )

        return self.output

    def backprop(self, delta_to_passbk, weight1):        
  
           delta_to_passbk= np.dot(delta_to_passbk, weight1.T) * sigmoid_derivative(self.weighted_sum)
           gradient = np.dot(layer.input.T, delta_to_passbk) 

           self.weights = self.weights -alpha * gradient
    
           return delta_to_passbk
    


x = np.array([ [0, 1], [1, 0], [1, 10],
               [1, 1],[0, 0], [9,9.0001], [8.8888, 8.8881] ])
y = np.array([ [1], [1], [1],
               [0], [0], [0], [0]    ])

layers=[]

num_input = 2
num_hidden = 5
num_output = 1

layer_h= Layer(num_input,num_hidden)
layer_i= Layer(num_hidden,num_hidden)
layer_y= Layer(num_hidden,num_output)

layers.append(layer_h)
layers.append(layer_i)
layers.append(layer_y)


def sigmoid(z):
    return 1 / (1+np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z)/((1+np.exp(-z))**2)


def cost_function(y, y_hat):
    J = 0.5*sum((y-y_hat)**2)
    
    return J

def update_last_layer():
    last_layer=layers[-1]
    y_hat = last_layer.output
    z2    = last_layer.weighted_sum
    
    delta_to_passbk = np.multiply(-(y-y_hat),sigmoid_derivative(z2))
    
    gradient_y = np.dot(last_layer.input.T, delta_to_passbk)
    last_layer.weights = last_layer.weights -alpha * gradient_y

    return delta_to_passbk,y_hat



alpha = 0.01
num_iterations = 5000

cost = []
for i in range(num_iterations):
    
    #perform forward propagation and predict output
    output=x
    for layer in layers:
        output=layer.forward_prop(output)

    """ 
    for layer in layers:
      print(layer.weights);
      print(layer.bias);
        
      print(layer.input);    
      print(layer.weighted_sum);
      print(layer.output);  
      print("\n-------------------------------------\n")
    """
    
    delta_to_passbk, y_hat= update_last_layer()
    
    for layer in reversed(layers[:-1]):
        idx       =layers.index(layer)
        weight1   = layers[idx+1].weights
        
        delta_to_passbk = layer.backprop(delta_to_passbk, weight1)
 
    
    #compute cost
    c = cost_function(y, y_hat)
    
    #store the cost
    cost.append(c)

print(cost[(num_iterations-10):])


