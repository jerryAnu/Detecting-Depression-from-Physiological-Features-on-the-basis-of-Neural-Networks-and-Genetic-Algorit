"""
This file is used to utilize genetic algorithm to find an optimal
learning rate for the model.
The fitness function is overall accuracy of the model.
"""


# import libraries
import numpy as np
import pandas as pd
import torch
import copy


"""
Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: 50 neurons, representing the physiological features of observers
    hidden layer: 60 neurons, using Sigmoid as activation function
    output layer: 4 neurons, representing various levels of depression

The network will be trained with Adam as an optimiser, 
that will hold the current state and will update the parameters
based on the computed gradients.

Its performance will be evaluated using cross-entropy.
"""
# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        """
        This function is to initialize the net model; specifically, we define
        a linear hidden layer and a linear output layer.
        :param n_input: the number of input neurons
        :param n_hidden: the number of hidden neurons
        :param n_output: the number of output neurons
        """
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
        This function is to define the process of performing forward pass,
        that is to accept a Variable of input data, x, and return a Variable
        of output data, y_pred.
        :param x: a variable of input data
        :return: a variable of output data
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # use dropout function to reduce the impact of overfitting
        h_input = torch.dropout(h_input, p=0.5, train=True)
        # define activation function for hidden layer
        h_output = torch.sigmoid(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred


def prec_reca_f1(confusion):
    """
    This function is used to calculate evaluation measures of this task
    including the precision, recall and F1-score.
    Precision: the number of true positive classifications divided by the number of
               true positive plus false positive classifications.
    Recall: the number of true positive classifications divided by the number of
            true positive plus false negative classifications.
    F1-score: 2 * Precision * Recall / (Precision + Recall)
    Besides, this function also calculates overall accuracy.
    Overall accuracy: the sum of correct predictions for all levels divided by
                      the number of data points.
    :param confusion: a confusion matrix
    :return: the precision, recall and F1-score for all levels of depression
             as well as overall accuracy
    """
    # calculate the numbers of true positive classifications, true positive
    # plus false positive classifications and true positive plus
    # false negative classifications for all four levels of depression.
    tp_none, tp_fp_none, tp_fn_none = confusion[0, 0], sum(confusion[:, 0]), sum(confusion[0, :])
    tp_mild, tp_fp_mild, tp_fn_mild = confusion[1, 1], sum(confusion[:, 1]), sum(confusion[1, :])
    tp_moderate, tp_fp_moderate, tp_fn_moderate = confusion[2, 2], sum(confusion[:, 2]), sum(confusion[2, :])
    tp_severe, tp_fp_severe, tp_fn_severe = confusion[3, 3], sum(confusion[:, 3]), sum(confusion[3, :])

    # avoid divided by zero and calculate the precision
    if tp_fp_none == 0:
        precision_none = 0
    else:
        precision_none = tp_none / tp_fp_none
    if tp_fp_mild == 0:
        precision_mild = 0
    else:
        precision_mild = tp_mild / tp_fp_mild
    if tp_fp_moderate == 0:
        precision_moderate = 0
    else:
        precision_moderate = tp_moderate / tp_fp_moderate
    if tp_fp_severe == 0:
        precision_severe = 0
    else:
        precision_severe = tp_severe / tp_fp_severe

    # calculate the average precision
    precision_average = (precision_none + precision_mild + precision_moderate + precision_severe) / 4

    # avoid divided by zero and calculate the recall
    if tp_fn_none == 0:
        recall_none = 0
    else:
        recall_none = tp_none / tp_fn_none
    if tp_fn_mild == 0:
        recall_mild = 0
    else:
        recall_mild = tp_mild / tp_fn_mild
    if tp_fn_moderate == 0:
        recall_moderate = 0
    else:
        recall_moderate = tp_moderate / tp_fn_moderate
    if tp_fn_severe == 0:
        recall_severe = 0
    else:
        recall_severe = tp_severe / tp_fn_severe

    # calculate the average recall
    recall_average = (recall_none + recall_mild + recall_moderate + recall_severe) / 4

    # avoid divided by zero and calculate the F1-score
    if precision_none == 0 and recall_none == 0:
        f1_score_none = 0
    else:
        f1_score_none = 2 * precision_none * recall_none / (precision_none + recall_none)
    if precision_mild == 0 and recall_mild == 0:
        f1_score_mild = 0
    else:
        f1_score_mild = 2 * precision_mild * recall_mild / (precision_mild + recall_mild)
    if precision_moderate == 0 and recall_moderate == 0:
        f1_score_moderate = 0
    else:
        f1_score_moderate = 2 * precision_moderate * recall_moderate / (precision_moderate + recall_moderate)
    if precision_severe == 0 and recall_severe == 0:
        f1_score_severe = 0
    else:
        f1_score_severe = 2 * precision_severe * recall_severe / (precision_severe + recall_severe)

    # calculate the average F1-score
    f1_score_average = (f1_score_none + f1_score_mild + f1_score_moderate + f1_score_severe) / 4

    # calculate overall accuracy
    overall_accuracy = (confusion[0, 0] + confusion[1, 1] + confusion[2, 2] + confusion[3, 3]) / (torch.sum(confusion))

    # combine the precision, recall, the F1-score and overall accuracy
    res = [[precision_none, precision_mild, precision_moderate, precision_severe],
           [recall_none, recall_mild, recall_moderate, recall_severe],
           [f1_score_none, f1_score_mild, f1_score_moderate, f1_score_severe],
           [precision_average, recall_average, f1_score_average, overall_accuracy]]

    return res


def overall_acc(confusion):
    """
    This function is to calculate overall accuracy.
    Overall accuracy: the sum of correct predictions for all levels divided by
                      the number of data points.
    :param confusion: a confusion matrix
    :return: overall accuracy
    """
    # calculate overall accuracy
    overall_accuracy = (confusion[0, 0] + confusion[1, 1] + confusion[2, 2] +
                        confusion[3, 3]) / (torch.sum(confusion))

    return overall_accuracy


def get_fitness(x_train, x_validation, lr=0.01):
    """
    This function is to calculate the fitness which is overall accuracy.
    This function trains and validates a network based on new features derived
    from GA; then calculate overall accuracy.
    :param x_train: training set
    :param x_validation: validation set
    :param lr: learning rate
    :return: evaluation measures
    """
    # create Tensors to hold inputs for training data
    X = x_train
    # create Tensors to hold inputs for validation data
    X_validation = x_validation

    # define the number of input neurons, hidden neurons, output neurons,
    # learning rate and training epochs
    input_neurons = X.shape[1]
    hidden_layer = 60
    output_neurons = 4
    learning_rate = lr
    num_epochs = 500

    # define a neural network using the customised structure
    net = TwoLayerNet(input_neurons, hidden_layer, output_neurons)

    # define loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # define optimiser
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # store all losses for visualisation
    all_losses = []

    # store the previous recorded overall accuracy
    previous_accuracy = 0
    # count how many times overall accuracy drops compared with
    # the previous recorded one.
    # if the value is more than 1, then stop training.
    count = 0
    # train a neural network
    for epoch in range(num_epochs):
        # perform forward pass: compute predicted y by passing x to the model.
        Y_pred = net(X)

        # compute loss
        loss = loss_func(Y_pred, Y)
        all_losses.append(loss.item())

        # print progress
        if epoch % 50 == 0:
            # use softmax function for classification
            Y_pred = torch.softmax(Y_pred, 1)
            # convert four-column predicted Y values to one column for comparison
            _, predicted = torch.max(Y_pred, 1)

            # create a confusion matrix, indicating for every level (rows)
            # which level the network guesses (columns).
            confusion = torch.zeros(output_neurons, output_neurons)
            # see how well the network performs on different categories
            for i in range(train_data.shape[0]):
                actual_class = Y.data[i]
                predicted_class = predicted.data[i]

                confusion[actual_class][predicted_class] += 1

            # calculate overall accuracy
            res = overall_acc(confusion)

            # if current overall accuracy is less than the previous one, "count"
            # is added one; otherwise, set "count" to zero and store current
            # overall accuracy
            # if "count" is more than 1, stop training
            if previous_accuracy >= res:
                # "count" is added one
                count += 1
                if count > 1:
                    # stop training
                    count = 0
                    break
            else:
                # otherwise, set "count" to zero
                count = 0
                # store current overall accuracy
                previous_accuracy = res

        # clear the gradients before running the backward pass.
        net.zero_grad()

        # perform backward pass
        loss.backward()

        # calling the step function on an Optimiser makes an update to its
        # parameters
        optimiser.step()

    """
    Evaluating the Results

    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every glass (rows)
    which class the network guesses (columns).

    """
    # create a confusion matrix
    confusion = torch.zeros(output_neurons, output_neurons)
    # perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)
    # convert four-column predicted Y values to one column for comparison
    _, predicted = torch.max(Y_pred, 1)

    # calculate the confusion matrix and print the matrix
    for i in range(train_data.shape[0]):
        actual_class = Y.data[i]
        predicted_class = predicted.data[i]

        confusion[actual_class][predicted_class] += 1

    """
    Validate the neural network
    
    Pass validation data to the built neural network and get its performance
    """
    # validate the neural network using validation data
    # It is actually performing a forward pass computation of predicted y
    # by passing x to the model.
    # Here, Y_pred_validation contains four columns
    Y_pred_validation = net(X_validation)

    # get prediction
    # convert four-column predicted Y values to one column for comparison
    _, predicted_validation = torch.max(Y_pred_validation, 1)

    """
    Evaluating the Results
    
    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every iris flower (rows)
    which class the network guesses (columns).
    
    """
    # create a confusion matrix
    confusion_validation = torch.zeros(output_neurons, output_neurons)

    # calculate the confusion matrix and print the matrix
    for i in range(validation_data.shape[0]):
        actual_class = Y_validation.data[i]
        predicted_class = predicted_validation.data[i]

        confusion_validation[actual_class][predicted_class] += 1

    # calculate overall accuracy
    res_validation = prec_reca_f1(confusion_validation)

    # return overall accuracy
    return res_validation


def roulette_wheel(pop, fitness):
    """
    This function is to select some chromosomes through roulette wheel selection.
    :param pop: population
    :param fitness: fitness
    :return: selected chromosomes
    """
    # copy fitness
    fitness_copy = copy.deepcopy(fitness)
    # calculate the total fitness
    total = sum(fitness_copy)
    # select chromosome based on each chromosome's fitness
    for i in range(len(fitness_copy)):
        fitness_copy[i] /= total
        if fitness_copy[i] == None:
            fitness_copy[i] = 0

    # implement roulette wheel selection
    idx = np.random.choice(np.arange(population_size), size=population_size, replace=True, p=fitness_copy)

    # return selected chromosomes
    return pop[idx]


def elitism(pop, fitness):
    """
    This function is to select some chromosomes through elitism selection.
    :param pop: population
    :param fitness: fitness
    :return: selected chromosomes
    """
    # copy fitness
    fitness_copy = copy.deepcopy(fitness)
    # find the best chromosome
    best_one = np.argmax(fitness_copy)
    # find the worst chromosome
    worst_one = np.argmin(fitness_copy)
    # store the best one in the first element
    tmp_pop = pop[0]
    tmp_fitness = fitness[0]
    pop[0] = pop[best_one]
    fitness[0] = fitness[best_one]
    pop[best_one] = tmp_pop
    fitness[best_one] = tmp_fitness
    # replace the worst one with the best one
    # the second element is also the best one
    tmp_pop = pop[1]
    tmp_fitness = fitness[1]
    pop[1] = pop[best_one]
    fitness[1] = fitness[best_one]
    pop[worst_one] = tmp_pop
    fitness[worst_one] = tmp_fitness
    # calculate the total fitness
    total = sum(fitness_copy)
    # select chromosome based on each chromosome's fitness
    for i in range(len(fitness_copy)):
        fitness_copy[i] /= total
        if fitness_copy[i] == None:
            fitness_copy[i] = 0

    idx = np.random.choice(np.arange(population_size), size=population_size, replace=True, p=fitness_copy)
    # keep the two best chromosomes
    idx[0:2] = [0, 1]

    # return selected chromosomes
    return pop[idx]


def crossover(parent, pop):
    """
    This function is to implement crossover.
    :param parent: a parent chromosome
    :param pop: population
    :return: a mated chromosome
    """
    if np.random.rand() < cross_over_rate:
        # randomly choose another chromosome from the population
        id = np.random.randint(0, population_size, size=1)
        # randomly choose the crossover points
        temp = np.random.randint(0, 2, size=chrom_size)
        pts = temp.astype(np.bool)
        # create a child
        parent[pts] = pop[id, pts]

    return parent


def mutation(child):
    """
    This function is to implement mutation.
    :param child: a child chromosome
    :return: a new chromosome
    """
    # randomly mutate
    for point in range(chrom_size):
        if np.random.rand() < mutation_rate:
            child[point] = 1 if child[point] == 0 else 0

    return child


if __name__ == "__main__":
    # load training set and validation set
    train_data = pd.read_csv("train.csv")
    validation_data = pd.read_csv("validation.csv")

    # the number of features
    n_features = train_data.shape[1] - 1

    # split training data into input and target
    # the first column is target; others are features
    train_input = train_data.iloc[:, 1:n_features + 1]
    train_target = train_data.iloc[:, 0]

    # normalise training data by columns
    for column in train_input:
        train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.min()) /
                                                                           (x.max() - x.min()))

    # split validation data into input and target
    # the first column is target; others are features
    validation_input = validation_data.iloc[:, 1:n_features + 1]
    validation_target = validation_data.iloc[:, 0]

    # normalise validation input data by columns
    for column in validation_input:
        validation_input[column] = validation_input.loc[:, [column]].apply(lambda x: (x - x.min()) /
                                                                         (x.max() - x.min()))

    # create Tensors to hold inputs and outputs for training data
    X = torch.Tensor(train_input.values).float()
    Y = torch.Tensor(train_target.values).long()

    # create Tensors to hold inputs and outputs for validation data
    X_validation = torch.Tensor(validation_input.values).float()
    Y_validation = torch.Tensor(validation_target.values).long()

    # the combination of the original 85 features
    # "Ture" indicates using the corresponding feature; "False" indicates not
    # using the corresponding feature
    choose = [True, True, True, True, True, False, True, True, True, True, False, False,
              True, True, True, True, False, False, False, False, False, False, False, False,
              True, True, False, False, False, False, False, True, False, True, True, False,
              False, True, True, True, True, False, True, False, True, True, True, False,
              True, False, True, True, False, True, False, False, True, True, True, True,
              True, False, True, True, True, True, False, True, True, True, False, False,
              False, True, True, True, False, True, False, True, False, True, True, True,
              False]

    # create Tensors to hold inputs for training data
    X = X[:, choose]
    # create Tensors to hold inputs for validation data
    X_validation = X_validation[:, choose]
    # the number of features
    n_features = X.shape[1]

    # parameters of genetic algorithm including the length of a chromosome,
    # the number of population, the rate of crossover, the rate of mutation
    # and the number of generation
    chrom_size = 14
    population_size = 20
    cross_over_rate = 0.8
    mutation_rate = 1 / chrom_size
    n_generation = 15

    # create initial population
    pop = np.random.randint(low=0, high=2, size=(population_size, chrom_size))

    # store fitness
    fitness = []

    # steps for GA
    for _ in range(n_generation):
        # store evaluation measures
        measures = [[] * 1 for _ in range(12)]
        for i in range(population_size):
            # store evaluation measures for each chromosome
            store = [[] * 1 for _ in range(16)]
            # run some times to avoid fluctuation
            for _ in range(5):
                # calculate the current learning rate
                learning_rate = 0.0
                for j in range(chrom_size):
                    learning_rate = learning_rate * 2.0 + pop[i][j] * 1.0

                learning_rate /= (2**14)
                tmp_res = get_fitness(X, X_validation, learning_rate)
                for j in range(4):
                    for k in range(4):
                        store[j * 4 + k].append(tmp_res[j][k])

            # store the average fitness
            fitness.append(np.average(store[-1]))
            # store other evaluation measures
            for j in range(12):
                measures[j].append(np.average(store[j]))

        # print the maximum fitness and the corresponding performances
        max_id = np.argmax(fitness).item()
        print("Most fitted chromosome: ", pop[max_id, :], " the most fitness: ", fitness[max_id] * 100)
        print(
            'Validation of Precision of None: %.2f %% Precision of Mild: %.2f %% Precision of Moderate: '
            '%.2f %% Precision of Severe: %.2f %%' % (100 * measures[0][max_id], 100 * measures[1][max_id],
                                     100 * measures[2][max_id], 100 * measures[3][max_id]))

        print('Validation of Recall of None: %.2f %% Recall of Mild: %.2f %% Recall of Moderate: %.2f %% Recall'
              ' of Severe: %.2f %%' % (100 * measures[4][max_id], 100 * measures[5][max_id],
                                       100 * measures[6][max_id], 100 * measures[7][max_id]))

        print(
            'Validation of F1 score of None: %.2f %% F1 score of Mild: %.2f %% F1 score of Moderate: %.2f %% F1 score'
            ' of Severe: %.2f %%' % (100 * measures[8][max_id], 100 * measures[9][max_id],
                                     100 * measures[10][max_id], 100 * measures[11][max_id]))

        # selection process by elitism selection
        pop = elitism(pop, fitness)
        # copy the population
        pop_copy = copy.deepcopy(pop)
        # crossover and mutation processes
        for j in range(2, len(pop)):
            child = crossover(pop[j], pop_copy)
            child = mutation(child)
            pop[j] = child

        # clear fitness
        fitness.clear()
