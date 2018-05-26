import numpy as np
import random as rnd
from math import log, exp
import pickle
import sys
import matplotlib.pyplot as plt
import logging
import time

import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)

# def show(image):
#     """
#     Render a given numpy.uint8 2D array of pixel data.
#     """
#     from matplotlib import pyplot
#     import matplotlib as mpl
#     fig = pyplot.figure()
#     ax = fig.add_subplot(1,1,1)
#     imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
#     imgplot.set_interpolation('nearest')
#     ax.xaxis.set_ticks_position('top')
#     ax.yaxis.set_ticks_position('left')
#     pyplot.show()

class ActivationSigmoid:
    def __call__(self, IN_VEC):
        return 1. / (1. + np.exp(-IN_VEC))
    def derivative(self, out):
        return (1.0 - out) * out
class ActivationSoftmax:
    def __call__(self, IN_VEC):
        denominator = sum([exp(v) for v in IN_VEC])
        return np.array([exp(v) / denominator for v in IN_VEC ])
    def derivative(self, out):
        raise Error("ERROR: you should not have gotten here ActivationSoftmax")
class ActivationInputIdentity:
    def __call__(self, IN_VEC):
        return IN_VEC
    def derivative(self, out):
        return np.array([.0,])

pic_size=28*28
nclasses=10
train_x = []
train_y = []
for label,img in read("training"):
    train_x.append(np.array([float(x) / 255 for x in img.reshape(-1)]))
    train_y.append(label)

test_x = []
test_y = []
for label,img in read("testing"):
    test_x.append(np.array([float(x) / 255 for x in img.reshape(-1)]))
    test_y.append(label)

# test_x = np.loadtxt("test_x")
architectures=[[pic_size,100,50,nclasses]]
epocs=[10,100,300]
learning_rates=[0.01,0.05]
weight_init_boundries=[0.08,0.5]

validation_ratio=.2
Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])
logging.basicConfig(filename="nn.log",level=logging.DEBUG)

def init_model(params):
    '''
    initialize the weight matrices using a uniformly distributed weight around 0,
    where each cell in row i and column j of weigh matrix l represents the incoming weights to neuron i in layer l+1
    from neuron j in layer l
    '''
    layer_sizes, weight_init_boundry = params
    return [ np.matrix([[rnd.uniform(-weight_init_boundry,weight_init_boundry) for i in range(layer_sizes[l])] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ], \
           [ np.matrix([rnd.uniform(-weight_init_boundry,weight_init_boundry) for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]
class LossNegLogLikelihood:
    '''
    used in order to callculate the validation
    '''
    def __call__(self, V, y):
        return -log(np.squeeze(np.asarray(V))[int(y)])
    def derivative_z(self, out, Y):
        return out-Y
loss=LossNegLogLikelihood()
def split_to_valid(train_x,train_y):
    data_set=zip(train_x, train_y)
    rnd.shuffle(data_set)
    train_size=len(data_set)-int(validation_ratio*len(data_set))
    return data_set[:train_size],data_set[:train_size]
sigmoid = ActivationSigmoid()
def fprop(W,B,X):
    '''
    forward propagate the input vector X through the 2 weight matrices and the weight vectors from the bias
    '''
    W1=W[0]
    b1=B[0]
    W2=W[1]
    b2=B[1]
    W3=W[2]
    b3=B[2]
    x=X
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1.T).reshape(-1) + b2
    h2 = sigmoid(z2)
    z3 = np.dot(W3, h2.T).reshape(-1) + b3
    h3 = sigmoid(z3)
    return [h1,h2,h3]
def bprop(W,B,X,Y,learning_rate):
    '''
    back propagate in order to calculate the error of each layer using the negative log likelyhood or
    the activation (sigmoid) derivative, and evevntually update the weight matrices accordingly
    '''
    out_list=fprop(W,B,X)
    h3 = out_list[2]
    h2 = out_list[1]
    h1 = out_list[0]
    y=np.matrix([Y])
    dz3 = (h3 - y)
    dW3 = np.outer(dz3, h2.T)
    db3 = dz3
    dz2 = np.array(np.dot(W[2].T, (h3 - y).T).reshape(-1).tolist()) * np.array(h2.tolist()) * (1. - np.array(h2.tolist()))
    dW2 = np.outer(dz2, h1.T)
    db2 = dz2
    dz1 = np.array(np.dot(W[1].T, dz2.T).reshape(-1).tolist()) * np.array(h1.tolist()) * (1. - np.array(h1.tolist()))
    dW1 = np.outer(dz1, X.T)
    db1 = dz1
    W[2] = W[2] - learning_rate*dW3
    B[2] = B[2] - learning_rate*db3
    W[1] = W[1] - learning_rate*dW2
    B[1] = B[1] - learning_rate*db2
    W[0] = W[0] - learning_rate*dW1
    B[0] = B[0] - learning_rate*db1
def validate(W,B,valid):
    '''
    validate that the average loss is descending and the accuracy is accending, in order to see the setup converges
    '''
    sum_loss= 0.0
    correct=0.0
    for X, y in valid:
        out = fprop(W,B,X)
        sum_loss += loss(out[-1],y)
        if out[-1].argmax() == y:
            correct += 1
    return sum_loss/ len(valid), correct/ len(valid)
def train(W,B,train_x,train_y,learning_rate,starting_epoc,ending_epoc,avg_loss_list,avg_acc_list):
    '''
    train for the given period of epocs, validating for convergance on each epoc
    (usefull for testing for the right hyper param configuration and architecture setup)
    '''
    train,valid=split_to_valid(train_x,train_y)
    for e in range(starting_epoc,ending_epoc):
        logging.debug("starting epoc {}".format(e))
        rnd.shuffle(train)
        s=time.time()
        for X,y in train:
            bprop(W,B,X,Y[y],learning_rate)
        duration=time.time()-s
        avg_loss,acc=validate(W, B, valid)
        logging.debug("epoc {} avg_loss {} acc {} duration {}".format(e,avg_loss,acc,duration))
        avg_loss_list.append(avg_loss)
        avg_acc_list.append(acc)
    epocs_list = list(range(ending_epoc))
    plt.plot(epocs_list,avg_loss_list,'red',epocs_list,avg_acc_list,'blue')
    plt.xlabel("epocs")
    plt.savefig("perf.e_{}.lr_{}.hs0_{}.hs1_{}.w_{}.png".format(ending_epoc,learning_rate,architectures[0][1],architectures[0][2],weight_init_boundries[0]))
    plt.clf()
    return avg_loss_list,avg_acc_list
def test(W,B,test_x,ending_epoc,learning_rate,architecture,weight_init_boundry):
    '''
    test over the test set using the learned weights matrix
    '''
    c=0.0
    with open("test.e_{}.lr_{}.hs1_{}.hs2_{}.w_{}.pred".format(
            ending_epoc,learning_rate,architecture[1],architecture[2],weight_init_boundry), 'w') as f:
        for X in test_x:
            p=np.squeeze(np.asarray(fprop(W, B, X)[-1]))
            # print("p {} y_hat {}".format(p,p.argmax()))
            f.write("{}\n".format(p.argmax()))

for weight_init_boundry in weight_init_boundries:
     for architecture in architectures:
         for learning_rate in learning_rates:
             W,B=init_model((architecture, weight_init_boundry))
             avg_loss_list = []
             avg_acc_list = []
             starting_epoc=0
             for ending_epoc in epocs:
                 logging.error("start training with params: e_{}.lr_{}.hs1_{}.hs2_{}.w_{}".format(ending_epoc,learning_rate,architecture[1],architecture[2],weight_init_boundry))
                 avg_loss_list, avg_acc_list = train(W,B,train_x,train_y,learning_rate,starting_epoc,ending_epoc,avg_loss_list,avg_acc_list)
                 starting_epoc = ending_epoc
                 # with open(r"W.e_{}.lr_{}.hs_{}.w_{}".format(ending_epoc,learning_rate,architecture[1],weight_init_boundry),"wb") as f:
                 #     f.write(pickle.dumps((W,avg_loss_list, avg_acc_list)))
                 test(W,B,test_x,ending_epoc,learning_rate,architecture,weight_init_boundry)