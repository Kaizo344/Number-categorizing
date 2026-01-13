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
hiddenSize = 256 #### <---------
outputSize = 10

## Input to output

W1 = np.random.randn(inputSize, hiddenSize) * np.sqrt(2.0 / inputSize)
W2 = np.random.randn(hiddenSize, outputSize) * np.sqrt(2.0 / hiddenSize) 

# W1 = np.random.randn(inputSize , hiddenSize) * 0.01
b1 = np.zeros((1,hiddenSize))
# print("weights", W1)
# print("outputs", B1)

## hidden to output layer
# W2 = np.random.randn(hiddenSize, outputSize) * 0.01
b2 = np.zeros((1,outputSize)) 

def reLU(num):
    return np.maximum(0,num)

def softMax(num):
    num = num - np.max(num, axis = 1, keepdims = True)
    exp_z = np.exp(num)
    return exp_z /np.sum(exp_z, axis = 1, keepdims = True)

def forwardPass(inputLayer):
    z1 = inputLayer @ W1 + b1
    a1 = reLU(z1)

    z2 = a1 @ W2 + b2
    a2 = softMax(z2)

    return z1, a1, z2, a2

### testing shapes

# x_Batch = XTrain[:32]

# Z1, A1, Z2, A2 = forwardPass(x_Batch)

# print('x_Batch shape test:',x_Batch.shape)
# print('A2 shape test:',A2.shape)
# print('A2 row 0 values test :', np.sum(A2[0]))

### cross entropy loss logic 
 
def crossEntropy(y_hat, y):
    return -np.mean(np.sum( y * np.log(y_hat + 1e-9), axis = 1))

### back propagation logic
## reLU 

def reLUDerrivative(z):
    return (z > 0).astype(float)

def backProp(X, y, z1, a1, z2, y_hat):
    m = X.shape[0]

    dZ2 = y_hat - y
    dW2 = (a1.T @ dZ2) / m

    db2 = np.sum(dZ2, axis = 0, keepdims = True) / m
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * reLUDerrivative(z1)

    dW1 = (X.T @ dZ1)/ m

    db1 = np.sum(dZ1, axis = 0, keepdims = True) / m

    return dW1, db1, dW2, db2


def updateParams(W1, b1, W2, b2, lr, dW1, db1, dW2, db2):
    # print("entering update params")
    

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    # print("exiting update params")

    return W1, b1, W2, b2



def accuracy(y_hat, y):
    predics = np.argmax(y_hat, axis = 1)
    labels = np.argmax(y, axis = 1)

    return np.mean(predics == labels)

def trainingStep(X, y,W1, b1, W2, b2, lr = 0.1):
    # print("entering training step")

    z1, a1, z2, y_hat = forwardPass(X)
    loss = crossEntropy(y_hat, y)
    acc = accuracy(y_hat, y)
    
    dW1, db1, dW2, db2 = backProp(X, y, z1, a1, z2, y_hat)
    W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, lr, dW1, db1, dW2, db2)
    # print("exiting training step")

    return W1, b1, W2, b2, loss, acc


### testing
batchSize = 32
XBatch = XTrain[:batchSize]
YBatch = YTrainO[:batchSize]

# W1, B1, W2, B2, loss = trainingStep(XBatch, YBatch, W1, B1, W2, B2, lr = 0.1)
# print(loss)

def predict(X):
    _, _, _, yHat = forwardPass(X)
    return np.argmax(yHat, axis = 1)

#### <----------------------------------- training loop
epochs = 100

W1_copy = W1.copy()

sampleIdx = 0 
sampleImage = XTest[sampleIdx : sampleIdx + 1]
sampleLabel = np.argmax(YTestO[sampleIdx])

### create a sample
for epoch in range(epochs):
    
    idx = np.random.choice(len(XTrain), batchSize, replace = False)
    XBatch = np.array(XTrain)[idx]
    YBatch = np.array(YTrainO)[idx]

    W1, b1, W2, b2, loss, acc = trainingStep(XBatch, YBatch, W1, b1, W2, b2, lr = 0.1)

    _, _, _, yHatTrain = forwardPass(XTrain)
    _, _, _, yHatTest = forwardPass(XTest)
    trainAcc = accuracy(yHatTrain, YTrainO)
    testAcc = accuracy(yHatTest, YTestO)

    _, _, _, y_hat_sample = forwardPass(sampleImage)
    pred = np.argmax(y_hat_sample)
    confidence = np.max(y_hat_sample)

    print(f"epoch = {epoch}  |loss = {loss}  |Test accuracy  = {testAcc:.5f}  | Train Accuracy = {trainAcc:.5f}  |Prediction: = {pred} |Actual: {sampleLabel}  |Confidence: {confidence}")


    #### printing each prediction
    
    
    # print("Weight change:", np.sum(np.abs(W1 - W1_copy)))


