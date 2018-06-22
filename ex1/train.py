"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from ev_alg_ex1_nn import validate
import logging

def train_and_score(network, B, valid,i):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """

#    logging.info(" validate")
    _,avg_acc,acc,_=validate(network, B, valid)
    #logging.info("net {} avg loss {} avg_acc {} acc {}".format(i,avg_loss,avg_acc, acc))


    return avg_acc,acc  # 1 is accuracy. 0 is loss.
