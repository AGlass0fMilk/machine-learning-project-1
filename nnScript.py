import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

# test
import time
import datetime

# This changes whether we're running test code or the actual neural network
RUNNING_TEST = False


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


#Testing
def printToFile(text):
    t = time.time()
    st = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
    f = open('out.txt', 'a+')
    f.write(st + ": " + text + '\n')
    f.close()

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-1.0 * z))# applies the sig function to every element in z


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Reduce the size of the images by deleting edges


    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def preprocess_small():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_sample.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     - feature selection"""


    mat = loadmat('./mnist_sample.mat')
        # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(4996, 784))
    validation_preprocess = np.zeros(shape=(1000, 784))
    test_preprocess = np.zeros(shape=(996, 784))
    train_label_preprocess = np.zeros(shape=(4996,))
    validation_label_preprocess = np.zeros(shape=(1000,))
    test_label_preprocess = np.zeros(shape=(996,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 100  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[100:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 100] = tup[tup_perm[0:100], :]
            validation_len += 100

            validation_label_preprocess[validation_label_len:validation_label_len + 100] = label
            validation_label_len += 100

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    print('preprocess done')


    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


    #Testing
    #print("Shape of w1: " + str(np.shape(w1)))
    #print("Shape of w2: " + str(np.shape(w2)))
    #print("number of input: " + str(n_input))
    #print("number of hidden: " + str(n_hidden))
    #print("number of classes: " + str(n_class))
    #print("Shape of training data: " + str(np.shape(training_data)))
    #print("Shape of training_label: " + str(np.shape(training_label.shape)))
    #print("Training label: " + str(training_label))
    #print("Lambda: " + str(lambdaval))
    #exit()

    #print("W1: " + str(w1))
    #print("W2: " + str(w2))
    #print("Training Data: " + str(training_data))
    #print("Training Label: " + str(training_label))

    #exit()

    # Your code here
    # Get predicted output values using current weights:
    # Also get hidden layer outputs
    outputs = nnGetOuputs(w1, w2, training_data)
    z = outputs[0]
    o = outputs[1]

    #print(str(z))
    #exit()

    # print(str(np.shape(o)))
    #print(str(o))
    #exit()

    # print(str(train_label))
    # print(str(np.shape(train_label)))

    num_of_examples = np.shape(training_label)[0]

    # Format labeled data into 1-of-K format
    y = np.zeros([num_of_examples, n_class])
    # Set the 0 to 1 in the index of the true label
    # Note that we must convert the double values in train_label to integers
    # So they can be used for indexing
    y[np.arange(num_of_examples), np.int_(training_label)] = 1
    #print(str(train_label[0]))
    #print(str(y[0]))
    #print(str(o[0]))
    #print(str((o-y)[0]))
    #print(str(np.shape(y)))

    #exit()

    # Now we can easily combine them all into J (easily... haha)
    # Should give us a vector of J for each training example
    Ji = -1 * np.sum((y*np.log(o)) + ((1-y)*np.log(1-o)), axis=1)
    #print(str(Ji))
    # And this will give us the whole sum, the objective function!
    J = (1/num_of_examples) * np.sum(Ji)

    # Add regularization term
    regularization = np.sum((w1**2)) + np.sum((w2**2))
    regularization = (lambdaval/(2*num_of_examples)) * regularization

    J = J + regularization

    #print(str(Ji))
    #print(str(np.shape(Ji)))
    #print("Objective Function Output: " + str(J))
    #printToFile("Objective Function Output: " + str(J))

    # Now we have to calculate the objective function gradient
    # 1.) Calcualte the derivative of Ji with respect to W2
    deltaL = o - y
    #print(str(o))
    #print(str(y))
    #print(str(deltaL))

    #print(str(np.shape(deltaL)))   # (4996, 10)
    #print(str(np.shape(z)))        # (4996, 51)
    #print(str(np.shape(w1)))  # (50, 785)
    #exit()

    # Remove bias term
    z_no_bias = z[:,:-1]    # shape = (4996, 50)
    w2_no_bias = w2[:,:-1]  # shape = (10, 50)
    #w1_no_bias = w1[:,1:]  # shape = (50, 784)

    # Some really complicated looking numpy math below...
    # Basically what it does is it calculates the dJ/dW for each example
    # What results is a Matrix [Num_Outputs]X[Num_Hidden] that has [Num_of_examples] pages (3rd dimension)
    # Each page is a dJ/dW for the training example. The 3rd dimension represents each training example

    # Confirmed to be working correctly by George ("confirmed"...)
    grad_w2 = np.zeros([num_of_examples, n_class, n_hidden+1]) #initialize empty output array
    for example in range(num_of_examples):
        grad_w2[example] = (deltaL[example]*np.reshape(z[example],[n_hidden+1, 1])).T

    #print(str(np.shape(grad_w2))) #(4996, 10, 50)

    #2.) Calcualte the derivative of Ji with respect to W1
    # Output will need to be shape [Num_examples]X[Num_hidden]X[Num_Input]
    # In testing this shape will be (4996, 50, 784)
    grad_w1 = np.zeros([num_of_examples, n_hidden, n_input+1])

    # Should work but is VERY slow
    '''for example in range(num_of_examples):
        for j in range(n_hidden):
            for p in range(n_input):
                innerSum = 0
                for l in range(n_class):
                    innerSum += deltaL[example, l] * w2[l, j]
                grad_w1[example, j, p] = (1 - z_no_bias[example, j]) * z_no_bias[example, j] * innerSum * training_data[example, p]'''

    # Append bias to training_data
    bias = np.ones([np.shape(training_data)[0], 1])
    x = np.append(training_data, bias, axis=1)

    # Seems to be correct
    for example in range(num_of_examples):
        innerSum = deltaL[example]*w2_no_bias.T
        innerSum = np.sum(innerSum, axis=1) #(4,)
        zTimes = (1-z_no_bias[example])*z_no_bias[example] #(4,)
        #print(np.shape(zTimes))
        grad_w1[example] = (zTimes * innerSum * np.reshape(x[example],[n_input+1,1])).T

    #print(str(np.shape(w1)))
    #print(str(np.shape(w2)))
    #print(str(np.shape(grad_w1))) #(4996, 50, 784)
    #print(str(np.shape(grad_w2))) #(4996, 10, 50)
    #print(str(grad_w1[0][0]))
    #exit()

    # Here we flatten the gradient matrices
    # This sums the gradients matrices for all training examples
    grad_w1 = np.sum(grad_w1, axis=0)
    grad_w2 = np.sum(grad_w2, axis=0)

    # Add regularization
    grad_w1 = grad_w1 + (lambdaval*w1)
    grad_w2 = grad_w2 + (lambdaval*w2)

    # Multiply each by 1/number of examples
    grad_w1 = (1/num_of_examples)*grad_w1
    grad_w2 = (1/num_of_examples)*grad_w2

    #print("Grad_w1: " + str(grad_w1))
    #print("Grad_w2: " + str(grad_w2))

    # Add the bias back on
    # THIS WAS THE SOURCE OF ALL WASTED TIME
    '''bias = np.ones([np.shape(grad_w1)[0], 1])
    grad_w1 = np.append(grad_w1, bias, axis=1)
    #grad_w1 = np.concatenate([bias,grad_w1], axis=1) #to prepend to first column
    bias = np.ones([np.shape(grad_w2)[0], 1])
    grad_w2 = np.append(grad_w2, bias, axis=1)'''
    #grad_w2 = np.concatenate([bias,grad_w2], axis=1)

    #print(str(np.shape(grad_w1))) #(50, 785)
    #print(str(np.shape(grad_w2))) #(10, 51)
    #print(str(grad_w1[0]))
    #print(str(grad_w2))
    #exit()

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val = J
    #obj_grad = np.array([])

    return (obj_val, obj_grad)

'''
This function returns a matrix of dimensions:
{[Number of rows in data]X[Number of hidden units],
[Number of rows in data]X[Number of output units]}
Effectively, this function returns a matrix that has
the raw floating-point output values of each hidden AND output node
for each training example

The output for hidden nodes is contained in return[0]
The output for output nodes is contained in return[1]
'''
def nnGetOuputs(w1, w2, data):
    # print(str(np.shape(w1)))
    # print(str(np.shape(w2)))

    # Add bias
    # print("Shape of train data: " + str(np.shape(data)))
    bias = np.ones([np.shape(data)[0], 1])
    # print(str(np.shape(bias)))
    #data = np.concatenate([bias,data], axis=1)
    data = np.append(data, bias, axis=1)
    # print("Shape of train data now: " + str(np.shape(data)))

    # The outputs of the hidden layer are the inner product of w1 with data (and transpose it)
    hiddenOut = np.inner(w1, data).T
    # Apply sigmoid activation function
    hiddenOut = sigmoid(hiddenOut)

    # print(str(np.shape(hiddenOut)))

    # Append a bias term
    bias = np.ones([np.shape(hiddenOut)[0], 1])
    #hiddenOut = np.concatenate([bias,hiddenOut], axis=1)
    hiddenOut = np.append(hiddenOut, bias, axis=1)

    # print(str(np.shape(hiddenOut)))

    # Compute inner product and apply sigmoid to get output
    output = np.inner(w2, hiddenOut).T
    output = sigmoid(output)

    # print(str(np.shape(output))) #This will be [Number of training examples] rows by [number of outputs] columns

    # We return both the hidden unit outputs and the final ouputs
    # This allows nnObjFunction to use both hidden and final outputs
    return [hiddenOut, output]

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    """
    Output: Should be a column vector with same number of rows as training data.
    (One output for each example).
    Output for each training set should be the label for the maximum value from the output nodes
    """

    output = nnGetOuputs(w1, w2, data)[1] #Get the ouputs from the other function

    #To get the single column of outputs, we need to take the maximum value from each row
    labels = np.argmax(output, axis=1)

    #print(str(np.shape(labels)))

    return labels

"""**************Testing Script Starts here***************************************"""
if(RUNNING_TEST):
    # Test Data
    n_input = 5
    n_hidden = 3
    n_class = 2
    training_data = np.array([np.linspace(0, 1, num=5), np.linspace(1, 0, num=5)])
    training_label = np.array([0, 1])
    lambdaval = 0
    params = np.linspace(-5, 5, num=26)
    args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)

    # Get output from nnObjFunction
    out = nnObjFunction(params, *args)
    #print(str(out[0]))
    print(str(out[1]))
    exit()


"""**************Neural Network Script Starts here********************************"""

startTime = time.time()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#  To switch between the large dataset
#  and the small dataset, comment out the
#  proper line below!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess_small()
#train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

#w1 = initial_w1
#w2 = initial_w2

#print(str(nnPredict(initial_w1, initial_w2, train_data)))
#exit()

# set the regularization hyper-parameter
lambdaval = 1

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
printToFile('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
printToFile('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
printToFile('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

elapsedTime = time.time() - startTime
print(elapsedTime)
