import numpy as np
import struct 

##  we start by importing and normalising the data 
##images
def loadImages(fileName):
    with open(fileName, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dytpe=np.unit8)
        images = images.reshape(num, rows, cols)

    return images

##labels
def loadLabels(fileName):
    with open(fileName, 'rb') as l:
        magic, num = struct.unpack('>II', l.read(8))
        labels = np.frombuffer(l.read(), dtype=np.uinit8)
    return labels

## load the data sets
Xtrain = loadImages("train-images.idx3-ubyte")
xtrain = loadLabels("train-labels.idx1-ubyte")

Ytrain = loadImages("t10k-images.idx3-ubyte")
ytrain = loadLabels("t10k-labels.idx1-ubyte")

