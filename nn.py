"""
This file is used to detect levels of depression by using a neural network model.
This model is a three-layer network.
This model is not combined with the GIS technique or genetic algorithm.
"""


# import libraries
import pandas as pd
import torch


"""
Define a neural network 

Here we build a neural network with one hidden layer.
    input layer: 85 neurons, representing the physiological features of observers
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


if __name__ == "__main__":
    # load training set and testing set
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")

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
    # This part is used to compare different methods of normalization
    # normalise for comparison (subtracted from the mean and divided by the standard deviation)
    # for column in train_input:
    #     train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

    # split testing data into input and target
    # the first column is target; others are features
    test_input = test_data.iloc[:, 1:n_features + 1]
    test_target = test_data.iloc[:, 0]

    # normalise testing input data by columns
    for column in test_input:
        test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.min()) /
                                                                         (x.max() - x.min()))
    # This part is used to compare different methods of normalization
    # normalise for comparison (subtracted from the mean and divided by the standard deviation)
    # for column in test_input:
    #     test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

    # create Tensors to hold inputs and outputs for training data
    X = torch.Tensor(train_input.values).float()
    Y = torch.Tensor(train_target.values).long()

    # create Tensors to hold inputs and outputs for testing data
    X_test = torch.Tensor(test_input.values).float()
    Y_test = torch.Tensor(test_target.values).long()

    # define the number of input neurons, hidden neurons, output neurons,
    # learning rate and training epochs
    input_neurons = n_features
    hidden_layer = 60
    output_neurons = 4
    learning_rate = 0.01
    num_epochs = 500

    # define a neural network using the customised structure
    net = TwoLayerNet(input_neurons, hidden_layer, output_neurons)

    # define loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # define optimiser
    optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # This part is used to compare different methods of optimiser
    # optimiser = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    # optimiser = torch.optim.Adagrad(net.parameters(), lr=learning_rate)

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

            # calculate evaluation measures and print the loss as well as
            # overall accuracy
            res = prec_reca_f1(confusion)

            print('Epoch [%d/%d] Loss: %.4f  Overall accuracy: %.2f %%'
                  % (epoch + 1, num_epochs, loss.item(), 100 * res[3][3]))

            # if current overall accuracy is less than the previous one, "count"
            # is added one; otherwise, set "count" to zero and store current
            # overall accuracy
            # if "count" is more than 1, stop training
            if previous_accuracy >= res[3][3]:
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
                previous_accuracy = res[3][3]

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
    create a confusion matrix, indicating for every level (rows)
    which level the network guesses (columns).
    
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

    print('')
    print('Confusion matrix for training:')
    print(confusion)

    # calculate evaluation measures and print the evaluation measures
    res_train = prec_reca_f1(confusion)

    print('Test of Precision of None: %.2f %% Precision of Mild: %.2f %% Precision of Moderate: %.2f %% Precision'
          ' of Severe: %.2f %%' % (100 * res_train[0][0], 100 * res_train[0][1],
                                   100 * res_train[0][2], 100 * res_train[0][3]))

    print('Test of Recall of None: %.2f %% Recall of Mild: %.2f %% Recall of Moderate: %.2f %% Recall'
          ' of Severe: %.2f %%' % (100 * res_train[1][0], 100 * res_train[1][1],
                                   100 * res_train[1][2], 100 * res_train[1][3]))

    print('Test of F1 score of None: %.2f %% F1 score of Mild: %.2f %% F1 score of Moderate: %.2f %% F1 score'
          ' of Severe: %.2f %%' % (100 * res_train[2][0], 100 * res_train[2][1],
                                   100 * res_train[2][2], 100 * res_train[2][3]))

    print('Average precision: %.2f %% Average recall: %.2f %% Average F1 score: %.2f %%'
          % (100 * res_train[3][0], 100 * res_train[3][1], 100 * res_train[3][2]))

    print('Overall accuracy: %.2f %%' % (100 * res_train[3][3]))

    """
    Test the neural network
    
    Pass testing data to the built neural network and get its performance
    """
    # test the neural network using testing data
    # It is actually performing a forward pass computation of predicted y
    # by passing x to the model.
    # Here, Y_pred_test contains four columns
    Y_pred_test = net(X_test)

    # get prediction
    # convert four-column predicted Y values to one column for comparison
    _, predicted_test = torch.max(Y_pred_test, 1)

    """
    Evaluating the Results
    
    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every iris flower (rows)
    which class the network guesses (columns).
    
    """
    # create a confusion matrix
    confusion_test = torch.zeros(output_neurons, output_neurons)

    # calculate the confusion matrix and print the matrix
    for i in range(test_data.shape[0]):
        actual_class = Y_test.data[i]
        predicted_class = predicted_test.data[i]

        confusion_test[actual_class][predicted_class] += 1

    print('')
    print('Confusion matrix for testing:')
    print(confusion_test)

    # calculate evaluation measures and print the evaluation measures
    res_test = prec_reca_f1(confusion_test)

    print('Test of Precision of None: %.2f %% Precision of Mild: %.2f %% Precision of Moderate: %.2f %% Precision'
          ' of Severe: %.2f %%' % (100 * res_test[0][0], 100 * res_test[0][1],
                                   100 * res_test[0][2], 100 * res_test[0][3]))

    print('Test of Recall of None: %.2f %% Recall of Mild: %.2f %% Recall of Moderate: %.2f %% Recall'
          ' of Severe: %.2f %%' % (100 * res_test[1][0], 100 * res_test[1][1], 100 * res_test[1][2],
                                   100 * res_test[1][3]))

    print('Test of F1 score of None: %.2f %% F1 score of Mild: %.2f %% F1 score of Moderate: %.2f %% F1 score'
          ' of Severe: %.2f %%' % (100 * res_test[2][0], 100 * res_test[2][1],
                                   100 * res_test[2][2], 100 * res_test[2][3]))

    print('Average precision: %.2f %% Average recall: %.2f %% Average F1 score: %.2f %%'
          % (100 * res_test[3][0], 100 * res_test[3][1], 100 * res_test[3][2]))

    print('Overall accuracy: %.2f %%' % (100 * res_test[3][3]))
