"""Entry point to evolving the neural network. Start here."""
import logging
from optimizer import Optimizer
from ev_alg_ex1_nn import pic_size
from ev_alg_ex1_nn import nclasses
from ev_alg_ex1_nn import test, test_x, train_x, train_y, test_y
import random as rnd
import matplotlib.pyplot as plt
import time

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)


logging.basicConfig(filename="C:\Users\\amir\Desktop\Keren\projects\ev_algo_ex1\data\\genetic.log",level=logging.INFO)
def train_networks(networks, valid):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """

    # logging.info(" split_to_valid")
    # nadav: marked out, used valid from the list instead
    #train,valid=split_to_valid(train_x,train_y)
    # logging.info("end split_to_valid")

    # s = time.time()
    i=1
    for network in networks:
        # s_ = time.time()
        network.train(valid,i)
        # logging.info("sinle net evaluation took {} sec".format(time.time() - s_))
        i+=1
    # logging.info("total net evaluation took {} sec".format(time.time()-s))

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.avg_accuracy

    return total_accuracy / len(networks)
num_of_chunks=10
def split_to_valid_chunks(train_x,train_y):
    data_set=zip(train_x, train_y)
    rnd.shuffle(data_set)
    valid_list=[]
    for i in range(num_of_chunks):
        valid_list.append(data_set[int(i*(1./num_of_chunks)*len(data_set)):int((i+1)*(1./num_of_chunks)*len(data_set))])
    #return data_set[:train_size],data_set[train_size:]
    return valid_list

def   generate(generations, population, nn_param_choices):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    optimizer = Optimizer(nn_param_choices)
    networks = optimizer.create_population(population)
    valid_list = split_to_valid_chunks(train_x,train_y)
    avg_acc_list = []
    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_networks(networks, valid_list[i%len(valid_list)])

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        avg_acc_list.append(average_accuracy)
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    epocs_list = list(range(generations))
    plt.plot(epocs_list,avg_acc_list,'red')
    plt.xlabel("generations")
    plt.ylabel("accuracy")
    plt.savefig("accuracy.png")
    plt.clf()

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])


    # test(networks[0].network, networks[0].B,([0,0],[0,1],[1,0],[1,1]))
    test(networks[0].network, networks[0].B,test_x,test_y)

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations =  1300 # Number of times to evole the population.
    population = 100  # Number of networks in each generation.

    nn_param_image = [pic_size,100,nclasses], 1

    nn_param = nn_param_image

    logging.info("Layer numbers {}, generations{}, population {}  ".format(nn_param[0],generations,population))

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, nn_param)

if __name__ == '__main__':
    main()
