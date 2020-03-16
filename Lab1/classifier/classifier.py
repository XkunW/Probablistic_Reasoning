import os.path
import numpy as np
import matplotlib.pyplot as plt
import util


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    # TODO: Write your code here
    # File lists
    spam_files = file_lists_by_category[0]
    ham_files = file_lists_by_category[1]
    # Target distributions
    pd = util.Counter()
    qd = util.Counter()
    # The number of times each word occurs in specific bag
    counts_in_spam = util.get_counts(spam_files)
    counts_in_ham = util.get_counts(ham_files)
    # SPAM bag size and HAM bag size
    spam_bag_size = sum(list(counts_in_spam.values()))
    ham_bag_size = sum(list(counts_in_ham.values()))
    # Dictionary
    dictionary = set(list(counts_in_spam.keys()) + list(counts_in_ham.keys()))
    # Assign distributions
    for word in dictionary:
        # A word can be either picked or not picked, hence 2
        pd[word] = (counts_in_spam[word] + 1) / (spam_bag_size + len(dictionary))
        qd[word] = (counts_in_ham[word] + 1) / (ham_bag_size + len(dictionary))
    """
    # Sanity Check
    
    s = 0
    for word in pd:
        s += pd[word]
    print("total pd: {}".format(s))

    s = 0
    for word in qd:
        s += qd[word]
    print("total qd: {}".format(s))
    """
    return pd, qd


def classify_new_email(filename, probabilities_by_category, prior_by_category, b=0):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    # TODO: Write your code here
    # 2 classes are handled the same way
    log_probabilities = [0, 0]
    x = util.get_words_in_file(filename)
    for i in range(2):
        for word in probabilities_by_category[i]:
            x_d = x.count(word)
            log_probabilities[i] += x_d * np.log(probabilities_by_category[i][word])
        log_probabilities[i] += prior_by_category[i]  # Since both 0.5, this line doesn't affect anything

    if log_probabilities[0] + b >= log_probabilities[1]:
        classify_result = ('spam', log_probabilities[0])
    else:
        classify_result = ('ham', log_probabilities[1])

    return classify_result


if __name__ == '__main__':

    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"
   
    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2, 2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 
    print("The following result does not include bias")
    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label, log_posterior = classify_new_email(filename,
                                                  probabilities_by_category,
                                                  priors_by_category)

        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0], totals[0], correct[1], totals[1]))

    # TODO: Write your code here to modify the decision rule such that
    # Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    # 0 Type 2 errors when bias is -9, 0 Type 1 errors when bias is 47
    print("The following result includes bias")

    type_1_error = []
    type_2_error = []

    for b in range(-38, 5, 3):
        performance_measures = np.zeros([2, 2])
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category,
                                                      b)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))
        type_1_error.append(totals[0] - correct[0])
        type_2_error.append(totals[1] - correct[1])

    np.savetxt('type_1_error.csv', type_1_error, delimiter=',')
    np.savetxt('type_2_error.csv', type_2_error, delimiter=',')

    plt.scatter(type_1_error, type_2_error, marker='*', s=150)
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    plt.title('Trade-off plot for SPAM Filter')
    plt.ylabel('Type 2 Error')
    plt.xlabel('Type 1 Error')

    plt.savefig('nbc.pdf')
    plt.show()
