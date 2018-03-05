# Language:       python3
# Creator:        Tevin Gladden
# Project:        Naive Bayes-Email spam binary classifier
#
# This classifier uses a data set of 15497 feature vectors, each with 335 features.
# Each instance is a representation of an email, where feature 0 is the classification(spam or non-spam)
# The remaining 334 features represent specific qualities of the email in question.
#
# Inputs:

# The user is prompted to enter a preffered test set size out of 15497 records.
# The proper proportion of the spam/non-spam records will be admitted into the training nad validation sets
#
# Outputs:
#
# The program provides sizes for the validation and training set after prompting for training set size
# The amount of correctly classified email samples
# The accuracy of the classification.

import numpy as np
import math, random

class NB:
    def __init__(self):
        self.spam = []
        self.non_spam = []
        self.training = []
        self.validation = []
        self.spam_prob = []
        self.non_spam_prob = []
        self.percent = 0.0

    def input(self):
        file1 = open("email.txt", mode = 'r')
        array1 = []

        for line in file1:
            array1 = line.strip("\n").split(" ")
            array1[2] = np.array(list(map(lambda x: int(x), array1[2])))
            if(array1[1] == '1'):
                self.spam.append(array1)
            elif(array1[1] == '-1'):
                self.non_spam.append(array1)
        random.shuffle(self.spam)               #records are shuffled to show importance of record/feature representation in our training set.
        random.shuffle(self.non_spam)

    def Set_Size(self):
        data_size = int(input("How many elements do you want in training set?\n"))
        spam_size = int(data_size*(float(len(self.spam)/(len(self.non_spam)+len(self.spam)))))
        non_spam_size = int(data_size*(float(len(self.non_spam)/(len(self.non_spam)+len(self.spam)))))

        self.training = (self.spam[:spam_size])
        self.training.extend(self.non_spam[:non_spam_size])
        self.validation = (self.spam[spam_size:])
        self.validation.extend(self.non_spam[non_spam_size:])

    def probability(self):
        length = (len(self.training[0][2]))
        count1 = np.zeros(length)
        count2 = np.zeros(length)

        for i in range(len(self.training)):
            if(self.training[i][1] == '1'):
                count1 = np.add(count1, self.training[i][2])
            else:
                count2 = np.add(count2, self.training[i][2])

        self.spam_prob = np.array(list(map(lambda x: x/float(length), count1)))
        self.non_spam_prob = np.array(list(map(lambda x: x/float(length), count2)))

    def predict(self):
        correct = 0
        for i in range(len(self.validation)):
            spam = np.dot(self.spam_prob, self.validation[i][2])
            non = np.dot(self.non_spam_prob, self.validation[i][2])

            if(spam > non and self.validation[i][1] == '1'):
                correct += 1
            elif(spam < non and self.validation[i][1] == '-1'):
                correct += 1

        percentage = 100*correct/float(len(self.validation))
        print("Validation Size:", len(self.validation))
        print("Training Size:", len(self.training))
        print("Correct:", correct, "of", len(self.validation))
        print("Prediction Percentage: %.2f" % (percentage))

    def out(self):
        print("spam", self.spam_prob)
        print("\n")
        print("non-spam", self.non_spam_prob)

    def API(self):
        self.input()
        self.Set_Size()
        self.probability()
        self.predict()

NB().API()