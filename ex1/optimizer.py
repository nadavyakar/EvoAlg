"""
Class that holds a genetic algorithm for evolving a network.

Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
from functools import reduce
from operator import add
import random
from network import Network
from ev_alg_ex1_nn import init_model
import numpy as np

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.2, mutate_chance=0.2):
        """Create an optimizer.

        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def creat_new_model(self):
        layer_sizes, weight_init_boundry = self.nn_param_choices
        return [ np.matrix([[0.0 for i in range(layer_sizes[l])] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ], \
           [ np.matrix([0.0 for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]


    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        """

        pop = []
        for _ in range(0, count):
            # Create a random network.
            W,B=init_model(self.nn_param_choices)

            network = Network(self.nn_param_choices)
            network.create_set(W,B)

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        child_1, child_1_B = self.creat_new_model()
        child_2, child_2_B  = self.creat_new_model()

        layers_size = len(mother.nn_param_choices[0])

        for layer_num in range(layers_size-1):
            weights_num = mother.nn_param_choices[0][layer_num+1]-1
            mother_weights = weights_num // 2
            #mother_random_layer_weights = np.random.randint(weights_num, size=mother_weights)
            mother_random_layer_weights = random.sample(range(weights_num), mother_weights)
            mother_layer = mother.network[layer_num]
            mother_b = mother.B[layer_num]
            father_layer = father.network[layer_num]
            father_b = father.B[layer_num]
            for weight in range(weights_num):

                if weight in mother_random_layer_weights:
                    child_1[layer_num][weight] += mother_layer[weight]
                    child_1_B[layer_num][0,weight] += mother_b[0,weight]
                    child_2[layer_num][weight] += father_layer[weight]
                    child_2_B[layer_num][0,weight] += father_b[0,weight]
                else:
                    child_1[layer_num][weight] += father_layer[weight]
                    child_1_B[layer_num][0,weight] += father_b[0,weight]
                    child_2[layer_num][weight] += mother_layer[weight]
                    child_2_B[layer_num][0,weight] += mother_b[0,weight]

         # Now create a network object.
        network_1 = Network(self.nn_param_choices)
        network_1.create_set(child_1, child_1_B)

        network_2 = Network(self.nn_param_choices)
        network_2.create_set(child_2, child_2_B)

        network_1 = self.mutate(network_1)
        network_2 = self.mutate(network_2)

        children.append(network_1)
        children.append(network_2)

        return children

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        # Choose a random key.


        layer_sizes = network.nn_param_choices[0]

        network_mutation = [ np.matrix([[np.random.normal(0,0.1) for i in range(layer_sizes[l])] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1)]
        B_mutation = [ np.matrix([np.random.normal(0,0.1) for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]

        for l in range(len(layer_sizes)-1):
            network.network[l] += network_mutation[l]
            network.B[l] += B_mutation[l]


        return network

    def   evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each network.

        graded = [(self.fitness(network), network) for network in pop]
        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        rank_sum = 0
        for i in range(len(graded)):
            rank_sum += graded[i].accuracy



        for i in range(len(graded)):
            net = graded[i]
            net.accuracy = float(net.accuracy) / rank_sum

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)
        ptential_parents_length = int(len(graded)*0.1)
        # The parents are every network we want to keep.

        parents = graded[:10]
        ptential_parents = graded[:ptential_parents_length]
        ptential_parents_lengh = len(ptential_parents)

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[10:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        #parents_length = len(parents)
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, ptential_parents_lengh-1)
            female = random.randint(0, ptential_parents_lengh-1)

            # Assuming they aren't the same network...
            if male != female:
                male = ptential_parents[male]
                female = ptential_parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
