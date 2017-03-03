import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

#test neural network

num_input = 3
num_hidden = 2
num_class = 2

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-1.0 * z))# applies the sig function to every element in z


def nnObjFunction(params, n_input, n_hidden, n_class, training_data, train_label, lambdaval):
    #n_input, n_hidden, n_class, training_data, training_label, lambdaval = myArgs

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Testing
    # print("Shape of w1: " + str(np.shape(w1)))
    # print("Shape of w2: " + str(np.shape(w2)))
    # print("number of input: " + str(n_input))
    # print("number of hidden: " + str(n_hidden))
    # print("number of classes: " + str(n_class))
    # print("Shape of training data: " + str(np.shape(training_data)))
    # print("Shape of training_label: " + str(np.shape(training_label.shape)))
    # print("Training label: " + str(training_label))
    # print("Lambda: " + str(lambdaval))
    #exit()

    # Your code here
    # Get predicted output values using current weights:
    # Also get hidden layer outputs
    outputs = nnGetOuputs(w1, w2, training_data)
    z = outputs[0]
    o = outputs[1]

    # print(str(np.shape(o)))
    #print(str(o))

    # print(str(train_label))
    # print(str(np.shape(train_label)))

    # Format labeled data into 1-of-K format
    num_of_examples = np.shape(train_label)[0]
    #y = np.zeros([num_of_examples, n_class])
    # Set the 0 to 1 in the index of the true label
    # Note that we must convert the double values in train_label to integers
    # So they can be used for indexing
    #y[np.arange(num_of_examples), np.int_(train_label)] = 1
    # print(str(y))
    # print(str(np.shape(y)))
    y = train_label

    # Now we can easily combine them all into J (easily... haha)
    # Should give us a vector of J for each training example
    #print(str(o))
    #print(str(y))
    Ji = -1 * np.sum((y * np.log(o)) + ((1 - y) * np.log(1 - o)), axis=1)
    # And this will give us the whole sum, the objective function!
    J = (1 / num_of_examples) * np.sum(Ji)

    #print(str(Ji))
    # print(str(np.shape(Ji)))
    print("Objective Function Output: " + str(J))

    # Now we have to calculate the objective function gradient
    # 1.) Calcualte the derivative of Ji with respect to W2
    deltaL = o - y

    # print(str(np.shape(deltaL)))   # (4996, 10)
    # print(str(np.shape(z)))        # (4996, 51)
    # print(str(np.shape(w1)))  # (50, 785)
    # exit()

    #print(str(deltaL))

    # Remove bias term
    z_no_bias = z[:, :-1]  # shape = (4996, 50)
    w2_no_bias = w2[:, :-1]  # shape = (10, 50)
    w1_no_bias = w1[:, :-1]  # shape = (50, 784)

    #print(str(z_no_bias))

    # Some really complicated looking numpy math below...
    # Basically what it does is it calculates the dJ/dW for each example
    # What results is a Matrix [Num_Outputs]X[Num_Hidden] that has [Num_of_examples] pages (3rd dimension)
    # Each page is a dJ/dW for the training example. The 3rd dimension represents each training example

    grad_w2 = np.zeros([num_of_examples, n_class, n_hidden])  # initialize empty output array
    for example in range(num_of_examples):
        grad_w2[example] = deltaL[example, :, None] * z_no_bias[example]

    # print(str(grad_w2))
    # print(str(np.shape(grad_w2))) #(4996, 10, 50)

    # 2.) Calcualte the derivative of Ji with respect to W1
    # Output will need to be shape [Num_examples]X[Num_hidden]X[Num_Input]
    # In testing this shape will be (4996, 50, 784)
    grad_w1 = np.zeros([num_of_examples, n_hidden, n_input])
    # innerSum = np.zeros([num_of_examples, n_class, n_hidden])
    # summation = np.zeros([num_of_examples, n_hidden])

    # exit()
    for example in range(num_of_examples):
        innerSum = deltaL[example, :, None] * w2_no_bias
        summation = np.sum(innerSum, axis=0)
        intermediate = (1 - z_no_bias[example]) * z_no_bias[example] * summation
        grad_w1[example] = (intermediate * training_data[example, :, None]).T

    # print(str(np.shape(grad_w1))) #(4996, 50, 784)
    # print(str(np.shape(grad_w2))) #(4996, 10, 50)

    print(str(grad_w1))

    # Here we flatten the gradient matrices
    # This sums the gradients matrices for all training examples
    grad_w1 = np.sum(grad_w1, axis=0)
    grad_w2 = np.sum(grad_w2, axis=0)

    # Multiply each by 1/number of examples
    grad_w1 = (1 / num_of_examples) * grad_w1
    grad_w2 = (1 / num_of_examples) * grad_w2

    # Add the bias back on
    bias = np.ones([np.shape(grad_w1)[0], 1])
    grad_w1 = np.concatenate([bias, grad_w1], axis=1)
    bias = np.ones([np.shape(grad_w2)[0], 1])
    grad_w2 = np.concatenate([bias, grad_w2], axis=1)

    # print(str(np.shape(grad_w1))) #(50, 785)
    # print(str(np.shape(grad_w2))) #(10, 51)
    # exit()

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    obj_val = J
    # obj_grad = np.array([])

    return (obj_val, obj_grad)

def nnGetOuputs(w1, w2, data):
    # print(str(np.shape(w1)))
    # print(str(np.shape(w2)))

    # Add bias
    # print("Shape of train data: " + str(np.shape(data)))
    bias = np.ones([np.shape(data)[0], 1])
    data = np.append(data, bias, axis=1)
    #print(str(data))

    # The outputs of the hidden layer are the inner product of w1 with data (and transpose it)
    hiddenOut = np.inner(w1, data).T
    # Apply sigmoid activation function
    hiddenOut = sigmoid(hiddenOut)
    #print(str(hiddenOut))

    # Append a bias term
    bias = np.ones([np.shape(hiddenOut)[0], 1])
    hiddenOut = np.append(hiddenOut, bias, axis=1)

    #print(str(hiddenOut))

    #exit()

    # Compute inner product and apply sigmoid to get output
    output = np.inner(w2, hiddenOut).T
    output = sigmoid(output)

    # print(str(np.shape(output))) #This will be [Number of training examples] rows by [number of outputs] columns

    # We return both the hidden unit outputs and the final ouputs
    # This allows nnObjFunction to use both hidden and final outputs
    return [hiddenOut, output]

def getGrad(dl, z, w2, x):
    num_of_examples = np.shape(z)[0]
    grad_w2 = np.zeros([num_of_examples, np.shape(dl)[1], np.shape(z)[1]])  # initialize empty output array
    #print(str(grad_w2))
    n_hidden = 4
    n_input = 2
    n_class = 2
    for example in range(num_of_examples):
        grad_w2[example] = (dl[example] * np.reshape(z[example], [n_hidden, 1])).T

    grad_w1 = np.zeros([num_of_examples, n_hidden, n_input])
    '''for example in range(num_of_examples):
        for j in range(n_hidden):
            for p in range(n_input):
                innerSum = 0
                for l in range(n_class):
                    innerSum += dl[example, l]*w2[l,j]
                grad_w1[example, j, p] = (1 - z[example, j]) * z[example, j] * innerSum * x[example,p]'''
    for example in range(num_of_examples):
        innerSum = dl[example]*w2.T
        innerSum = np.sum(innerSum, axis=1)
        zTimes = (1-z)*z
        grad_w1[example] = (zTimes * innerSum * x.T).T



            #print(str(grad_w1[example,0]))

    print(str(grad_w1))
    #print(str(grad_w2))
    return grad_w2

w1 = np.array([[0.1, 0.8, 0.2, 0.3],
               [0.4, 0.6, 0.3, 0.2]])

w2 = np.array([[0.3, 0.9, 0.45],
               [0.6, 0.4, 0.6]])

X = np.array([[0.35, 0.9, 0.7],
              [0.6, 0.5, 0.35]])

Y = np.array([[0.4, 0.7],
             [0.3, 0.6]])

#outputs = nnGetOuputs(w1, w2, X)

lambd = 1
daArgs = (num_input, num_hidden, num_class, X, Y, lambd)
weights = np.concatenate((w1.flatten(), w2.flatten()), 0)

dl = np.array([[-0.2, -0.3]])
              # [-0.2, -0.3]])
z = np.array([[0.4, 0.6, 0.3, 0.7]])
              #[0.4, 0.5, 0.3, 0.7]])

w2 = np.array([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8]])
x = np.array([[0.2, 0.6]])

getGrad(dl,z, w2, x)

#nnObjFunction(weights, num_input, num_hidden, num_class, X, Y, lambd)

