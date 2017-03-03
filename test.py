import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random


# import nnScript as nn

def testPreprocess():
    mat = loadmat('./mnist_all.mat')

    train9 = mat.get('train9')

    numOfSamples = 5

    s = random.sample(range(train9.shape[0]), numOfSamples)

    fig = plt.figure(figsize=(4, 4))

    numRowsCols = 4 #must be even

    for i in range(numOfSamples):
        plt.subplot(2, numOfSamples, i + 1)
        row = train9[s[i], :]
        # note that each row is a flattened image
        # we first reshape it to a 28x28 matrix
        reshaped = np.reshape(row, ((28, 28)))
        plt.imshow(reshaped)
        plt.axis('off')
        plt.subplot(2, numOfSamples, numOfSamples + i + 1)

        resized = np.reshape(row, [28,28])
        resized = resized[numRowsCols:, int(numRowsCols/2):-int(numRowsCols/2)]
        resized = resized.flatten()
        modified = np.reshape(resized, [28-numRowsCols, 28-numRowsCols])
        plt.imshow(modified)
        plt.axis('off')
    plt.show()

testPreprocess()


# nn.preprocess()

# exit()
