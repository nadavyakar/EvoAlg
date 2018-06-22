import numpy as np
import random as rnd
from math import log, exp
import pickle
import sys
import matplotlib.pyplot as plt
import logging
import os
import struct
import numpy as np
import math
import time
pic_size=784
nclasses=10
batch_size=1
validation_ratio=.1
epocs=[50,100,200,300] #nuber of learning iterations
learning_rates=[0.05,1]
architectures=[[pic_size,10,100,nclasses],[pic_size,100,100,nclasses],[pic_size,400,100,nclasses]] #layer_sizes
weight_init_boundries=[0.08,0.5] # weight boundries


Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ]) ## {0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
rnd.seed(1) #  the same set of numbers

# logger=logging.getLogger(__name__)
#logging.basicConfig(filename="C:\Users\\amir\Desktop\Keren\projects\ev_algo_ex1\data\\nn.log",level=logging.ERROR)

def read(dataset = "training", path = "."):
    #reads the DB and return image and label
    #
    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

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




def init_model(params):
    '''
    initialize the weight matrices using a uniformly distributed weight around 0,
    where each cell in row i and column j of weigh matrix l represents the incoming weights to neuron i in layer l+1
    from neuron j in layer l
    '''

    layer_sizes, weight_init_boundry = params
    W =  [ np.matrix([[np.random.normal(0, np.sqrt(6.0 / (layer_sizes[l] + layer_sizes[l+1]))) for i in range(layer_sizes[l])] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]
    B =  [ np.matrix([np.random.normal(0, np.sqrt(6.0 / (layer_sizes[l] + layer_sizes[l+1]))) for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]

    return W, B
def init_model2(params):
    '''
    initialize the weight matrices using a uniformly distributed weight around 0,
    where each cell in row i and column j of weigh matrix l represents the incoming weights to neuron i in layer l+1
    from neuron j in layer l
    '''
    layer_sizes, weight_init_boundry = params


    W = [ np.matrix([[0.0 for i in range(layer_sizes[l])] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]
    for l in range(len(layer_sizes)-1):
        for j in range(layer_sizes[l+1]):
            eps = np.sqrt(6.0/ (layer_sizes[l],layer_sizes[l+1]))
            #W[l,j] +=  np.matrix([[0.0 for i in range(layer_sizes[l])







#Sigmoid
class ActivationSigmoid:
    def __call__(self, IN_VEC):
        return 1. / (1. + np.exp(-IN_VEC))
    def derivative(self, out):
        return (1.0 - out) * out

#Relu
class ActivationRelu:
    def __call__(self, IN_VEC):
        return np.maximum(IN_VEC, 0.0)
    def derivative(self, out):
        return (1.0 - out) * out

#Softmax
class ActivationSoftmax:
    def __call__(self, IN_VEC):
        V = np.squeeze(np.array(IN_VEC))
        max_=max(V)
        denominator = sum([exp(v-max_) for v in V])
        return np.array([exp(v-max_) / denominator for v in V ])
    def derivative(self, out):
        raise Error("ERROR: you should not have gotten here ActivationSoftmax")

#???
class ActivationInputIdentity:
    def __call__(self, IN_VEC):
        return IN_VEC
    def derivative(self, out):
        return np.array([.0,])

#activation functions
activation=[ActivationInputIdentity(), ActivationSigmoid(), ActivationSigmoid(), ActivationSoftmax()]


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
    #return data_set[:train_size],data_set[train_size:]
    return data_set[:train_size],data_set[59900:]


sigmoid=ActivationSigmoid()
softmax=ActivationSoftmax()
relu=ActivationRelu()
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
    #h1 = relu(z1)
    z2 = np.dot(W2, h1.T).reshape(-1) + b2
    h2 = sigmoid(z2)
    #h2 = relu(z2)
    z3 = np.dot(W3, h2.T).reshape(-1) + b3
    h3 = softmax(z3)
    return [h1,h2,h3]

def fprop_relu(W,B,X):
    W1=W[0]
    b1=B[0]
    W2=W[1]
    b2=B[1]
    x=X
    z1 = np.dot(W1, x) + b1
    h1 = relu(z1)
    z2 = np.dot(W2, h1.T).reshape(-1) + b2
    h2 = softmax(z2)
    return [h1,h2]

def fprop_relu_no_softmax(W,B,X):
    W1=W[0]
    b1=B[0]
    W2=W[1]
    b2=B[1]
    W3=W[2]
    b3=B[2]
    x=X
    z1 = np.dot(W1, x) + b1
    # h1 = sigmoid(z1)
    h1 = relu(z1)
    z2 = np.dot(W2, h1.T).reshape(-1) + b2
    # h2 = sigmoid(z2)
    h2 = relu(z2)
    z3 = np.dot(W3, h2.T).reshape(-1) + b3
    h3 = relu(z3)
    return [h1,h2,h3]

def validate(W,B,valid):
    '''
    validate that the average loss is descending and the accuracy is accending, in order to see the setup converges
    '''
    sum_loss= 0.0
    correct=0.0
    i=0
    # s=time.time()
    out_list=[]
    for X, y in valid:
        s_=time.time()
        i+=1
        # out = fprop(W,B,X)
        out=fprop_relu(W,B,X)[-1].argmax()
        out_list.append(out)
        # v = loss(out[-1],y)
        # sum_loss += loss(out[-1],y)
      #  print("{} X  p {} y {}".format(i,out[-1].argmax(),y))
       # logging.error("{} X p {} y {}".format(i,out[-1].argmax(),y))
        if out == y:
            correct += 1
        # if i==1:
        #     logging.info("single net valid over first example took {} sec".format(time.time() - s_))
        #     s_=time.time()
        #     fprop_relu(W,B,X)
        #     logging.info("single net fprop using relu took {} sec".format(time.time()-s_))
        #     s_ = time.time()
        #     fprop_relu_no_softmax(W, B, X)
        #     logging.info("single net fprop using relu without softmax took {} sec".format(time.time() - s_))
    # logging.info("single net valid took {} sec for {} examples".format(time.time()-s,len(valid)))
    return sum_loss/ len(valid), correct/ len(valid), correct, out_list



def test(W,B,test_x,test_y):
    '''
    test over the test set using the learned weights matrix
    '''
    c=0.0
    _, avg_acc, _, out_list = validate(W, B, zip(test_x, test_y))
    logging.info("test avg acc: {}".format(avg_acc))
    with open("test.pred", 'w') as f:
        for out in out_list:
            f.write("{}\n".format(out))

train_x = []
train_y = []

for label,img in read("training", "."):
     train_x.append(np.array([float(x) / 255 for x in img.reshape(-1)]))
     train_y.append(label)
#train_x = train_x[:1000]
#train_y = train_y[:1000]

test_x = []
test_y = []
for label,img in read("testing", "."):
     test_x.append(np.array([float(x) / 255 for x in img.reshape(-1)]))
     test_y.append(label)


# data_set=zip(train_x, train_y)
# rnd.shuffle(data_set)

#for weight_init_boundry in weight_init_boundries:
#   for layer_sizes in architectures:
#      for learning_rate in learning_rates:
#          params=layer_sizes, weight_init_boundry
#         W = init_model(params)
#        avg_loss_list = []
#            avg_acc_list = []
#            starting_epoc=0
#            for ending_epoc in epocs:
#                params = starting_epoc, ending_epoc, learning_rate, layer_sizes, batch_size, weight_init_boundry, avg_loss_list, avg_acc_list
#                logging.error("start training with params: e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry))
#                W, avg_loss_list, avg_acc_list =train(W,train_x,train_y,params)
#                starting_epoc = ending_epoc
#                with open(r"C:\Users\\amir\Desktop\Keren\projects\ev_algo_ex1\data\W.e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry),"wb") as f:
#                    f.write(pickle.dumps((W,avg_loss_list, avg_acc_list)))
#                #pW=open(r"/home/nadav/data/W.e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(epocs,learning_rate,layer_sizes[1],batch_size,weight_init_boundry)).read()
#                #W,avg_loss_list, avg_acc_list=pickle.loads(pW)
#                test_and_write(W,test_x,params)
