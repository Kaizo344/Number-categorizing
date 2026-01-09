import numpy as np
import struct 

###  we start by importing and normalising the data 
### images
def loadImages(fileName):
    with open(fileName, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)

    return images

###labels
def loadLabels(fileName):
    with open(fileName, 'rb') as l:
        magic, num = struct.unpack('>II', l.read(8))
        labels = np.frombuffer(l.read(), dtype=np.uint8)
    return labels

### load the data sets
XTrain = loadImages("train-images.idx3-ubyte")
YTrain = loadLabels("train-labels.idx1-ubyte")

XTest = loadImages("t10k-images.idx3-ubyte")
YTest= loadLabels("t10k-labels.idx1-ubyte")

### data preparation

## normalise the data into a proportion
XTrain = XTrain.reshape(XTrain.shape[0], -1).astype(np.float32)/255.0
XTest  = XTest.reshape(XTest.shape[0], -1).astype(np.float32)/255.0

# print("Training data: ")
# print(XTrain.shape)
# print("Testing data: ")
# print(XTest.shape)


## one hot encoding 
def oneHot(y, numclasses = 10):
    return np.eye(numclasses)[y]

YTrainO = oneHot(YTrain)
YTestO = oneHot(YTest)

### forward pass

inputSize = 784
hiddenSize = 128
outputSize = 10

##input
W1 = np.random.randn(inputSize , hiddenSize) * 0.01
B1 = np.zeros((1,hiddenSize))

##output
W2 = np.random.randn(inputSize, hiddenSize) * 0.01
B2 = np.random.randn((1,outputSize)) 


