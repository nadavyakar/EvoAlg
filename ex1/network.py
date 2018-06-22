"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score

class Network():
    """Represent a network and let us operate on it.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
        """
        self.accuracy = 0.
        self.avg_accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}
        self.B = {}

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network, B):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network
        self.B = B

    def train(self,valid,i):
        """Train the network and record the accuracy.
        """


        if self.accuracy == 0.:
            self.avg_accuracy, self.accuracy = train_and_score(self.network, self.B,valid,i)

    def print_network(self):
        """Print out a network."""
        logging.info("Network accuracy: %.2f%%" % (self.avg_accuracy))
