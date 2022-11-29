import numpy as np
import matplotlib.pyplot as plt
rg = np.random.default_rng(20)

"""Define hyperparameters before building a model with the nn_model function."""

def initialize_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- iterable containing the dimensions of each layer, including
                  the input layer, in the neural network.
    
    Returns:
    parameters -- python dictionary containing model parameters "W1", "b1", ..., "WL", "bL"
    """
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        r, c = layer_dims[l], layer_dims[l-1]
        parameters['W%d' % l] = rg.normal(size=(r, c), scale=1/np.sqrt(c))
        parameters['b%d' % l] = np.zeros((r, 1))
    return parameters

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def forward_propagate(A, W, b, g):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    g -- activation function

    Returns:
    A -- activation for next layer 
    """
    Z = np.dot(W, A) + b
    return g(Z), Z

def compute_cost(AL, Y, m):
    return (-1/m) * np.sum((Y * np.log(AL)) + ((1-Y) * np.log(1-AL)))

def sigmoid_derivative(Z):
    A = sigmoid(Z)
    return A * (1 - A)

def relu_derivative(Z):
    return Z > 0

def tanh_derivative(z):
    a = np.tanh(z)
    return 1 - np.power(a, 2)

def get_weights_for_layer(l, parameters):
    W = parameters['W' + str(l)]
    b = parameters['b' + str(l)]
    return W, b

def get_activation_function_for_layer(l, hyperparameters):
    activation_functions = hyperparameters.get('activation_functions')
    relu = activation_functions.get('relu')
    sigmoid = activation_functions.get('sigmoid')
    return sigmoid if l == hyperparameters.get('L') else relu

def back_propagate(dA_next, A_prev, Z, W, l, L, m, parameters):
    derivative = sigmoid_derivative if l == L else relu_derivative
    g_p = derivative(Z)
    
    dZ = dA_next * g_p
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    parameters['dW' + str(l)] = dW
    parameters['db' + str(l)] = db
    
    if l != 1:
        return np.dot(W.T, dZ)

def propagate(A_prev, Y, l, parameters, hyperparameters):
    """Recursive implementation of forward and backward propagation."""
    L = hyperparameters.get('L')
    m = np.shape(Y)[1]
    if l == L+1:
        parameters['J'] = compute_cost(A_prev, Y, m)
        dA = np.divide(-Y, A_prev) + np.divide(1-Y, 1-A_prev)
        return dA
    else:
        W, b = get_weights_for_layer(l, parameters)
        g = get_activation_function_for_layer(l, hyperparameters)
        A, Z = forward_propagate(A_prev, W, b, g)
        dA = propagate(A, Y, l+1, parameters, hyperparameters)
    return back_propagate(dA, A_prev, Z, W, l, L, m, parameters)

def update(parameters, hyperparameters):
    L = hyperparameters['L']
    alpha = hyperparameters['learning_rate']
    for l in range(1, L+1):
        parameters['W' + str(l)] -= np.multiply(alpha, parameters['dW' + str(l)])
        parameters['b' + str(l)] -= np.multiply(alpha, parameters['db' + str(l)])

def optimize(A, Y, parameters, hyperparameters):
    costs = list()
    for t in range(hyperparameters.get('n_iterations')):
        propagate(A, Y, 1, parameters, hyperparameters)
        update(parameters, hyperparameters)
        if t % 100 == 0:
            costs.append(parameters.get('J'))
    return parameters, costs

def plot_cost(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
def nn_model(A, Y, hyperparameters):
    parameters = initialize_parameters(hyperparameters.get('nn_structure'))
    parameters, costs = optimize(A, Y, parameters, hyperparameters)
    #plot_cost(costs, hyperparameters.get('learning_rate'))
    return parameters

def get_weights_and_biases(parameters, hyperparameters):
    L = hyperparameters.get('L')
    return [(parameters.get('W%d' % l), parameters.get('b%d' % l)) for l in range(1, L+1)]

def predict(parameters, hyperparameters, A, y):
    w_and_b = get_weights_and_biases(parameters, hyperparameters)
    *hidden_layer_w_and_b, last_layer_w_and_b = w_and_b
    
    for W, b in hidden_layer_w_and_b:
        A = relu(np.dot(W, A) + b)
    W, b = last_layer_w_and_b
    
    y_hat = sigmoid(np.dot(W, A) + b)
    pred = (y_hat >= 0.5)
    accuracy = np.mean(pred == y)
    return y_hat, pred, accuracy

# A set of hyperparameters for getting started
hyperparameters = {

    'learning_rate': 0.002,
    'nn_structure': (2, 4, 4, 1),
    'n_iterations': 5_000,
    'activation_functions': {'relu': relu, 'sigmoid': sigmoid},
    'L': 3,
    'lambd': 0.1,
    
}