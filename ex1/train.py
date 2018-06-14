"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import numpy as np
import random as rnd
from ev_alg_ex1_nn import activation
from ev_alg_ex1_nn import split_to_valid
from ev_alg_ex1_nn import validate
import logging
from ev_alg_ex1_nn import test
from ev_alg_ex1_nn import plt
from ev_alg_ex1_nn import Y
from ev_alg_ex1_nn import architectures
from ev_alg_ex1_nn import weight_init_boundries
from ev_alg_ex1_nn import init_model
from ev_alg_ex1_nn import epocs
from ev_alg_ex1_nn import train_x
from ev_alg_ex1_nn import train_y
import pickle
from ev_alg_ex1_nn import weight_init_boundries
from ev_alg_ex1_nn import architectures
from ev_alg_ex1_nn import learning_rates


def train_and_score(network, B):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    avg_loss_list = []
    avg_acc_list = []




   # train_x_xor = ([0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1])
   # train_y_xor = (0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0)
    # train_xor,valid_xor=split_to_valid(train_x_xor,train_y_xor)


    #data_set_xor=zip(train_x_xor, train_y_xor)
    #rnd.shuffle(data_set_xor)

    #train_x_and = ([0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1])
    #train_y_and = (0,0,0,1, 0,0,0,1,0,0,0,1, 0,0,0,1)

    #data_set_and=zip(train_x_and, train_y_and)
    #rnd.shuffle(data_set_and)

    #data_set=zip(train_x, train_y)
    #rnd.shuffle(data_set)
    #avg_loss,acc=validate(network, B, data_set)

    logging.info(" split_to_valid")
    train,valid=split_to_valid(train_x,train_y)
    logging.info(" validate")
    avg_loss,acc=validate(network, B, valid)
    #avg_loss,acc=validate(network, B, valid)


    #avg_loss,acc=validate(network, B, data_set_and)
    #avg_loss,acc=validate(network, B, data_set_and)
   # print("acc {}".format(acc))
   # test(network, B,([0,0],[0,1],[1,0],[1,1]))

    #logging.error("acc {}".format(acc))
    logging.info(" avg loss {} accuracy {}".format(avg_loss,acc))


    return acc  # 1 is accuracy. 0 is loss.


def train_and_score2(network, B, valid,i):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    avg_loss_list = []
    avg_acc_list = []




    # train_x_xor = ([0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1])
    # train_y_xor = (0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0)
    # train_xor,valid_xor=split_to_valid(train_x_xor,train_y_xor)


    #data_set_xor=zip(train_x_xor, train_y_xor)
    #rnd.shuffle(data_set_xor)

    #train_x_and = ([0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1],[0,0],[0,1],[1,0],[1,1])
    #train_y_and = (0,0,0,1, 0,0,0,1,0,0,0,1, 0,0,0,1)

    #data_set_and=zip(train_x_and, train_y_and)
    #rnd.shuffle(data_set_and)

    #data_set=zip(train_x, train_y)
    #rnd.shuffle(data_set)
    #avg_loss,acc=validate(network, B, data_set)


    logging.info(" validate")
    avg_loss,acc=validate(network, B, valid)
    #avg_loss,acc=validate(network, B, valid)


    #avg_loss,acc=validate(network, B, data_set_and)
    #avg_loss,acc=validate(network, B, data_set_and)
    # print("acc {}".format(acc))
    # test(network, B,([0,0],[0,1],[1,0],[1,1]))

    #logging.error("acc {}".format(acc))
    logging.info("net {} avg loss {} accuracy {}".format(i,avg_loss,acc))


    return acc  # 1 is accuracy. 0 is loss.
