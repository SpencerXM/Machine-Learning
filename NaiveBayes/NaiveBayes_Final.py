###########################################################################
# naive bayesian for mix  (numeric variables and binary variables)
# author: Xinpei Ma
# date: 6/23/2017
# Parameters
# -----------
# specific column indexes of binary variables
#
###########################################################################

from __future__ import division
import csv
import random
import math
import numpy as np



class NativeBayes(object):
    def __init__(self, integerIndexList, ifMix = True):
        """ Creates a new Naive Bayes Model for Data have both categorical variables and numeric variables. """
        # You need to define which columns are categorical variables
        self.integerIndexList = integerIndexList
        self.ifMix = ifMix
        self.summaries = {}
        self.prior = {}
        self.posterior = {}

    def summarize(self, dataset):
        summaries = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dataset)]
        return summaries

    def fit(self, X, y):
        separated = {}
        for i in xrange(len(y)):
            vector = X[i]
            if y[i] not in separated:
                separated[y[i]] = []
            separated[y[i]].append(list(vector))


        for classValue, instances in separated.iteritems():
            self.summaries[classValue] = self.summarize(instances)


        distinctClassLabels = list(set(y))
        for i in range(len(distinctClassLabels)):
            self.prior[distinctClassLabels[i]] = len([j for j in y if j == distinctClassLabels[i]]) / len(y)


        for j in self.integerIndexList:
            alpha = len(list(set([item[j] for item in X])))
            posterior_probability = {}
            terms = list(set([item[j] for item in X]))
            for m in terms:
                for n in distinctClassLabels:
                    numerator = len([X[l][j] for l in xrange(len(X)) if X[l][j] == m and y[l] == n]) + 1
                    denominator = len([X[l][j] for l in xrange(len(X)) if X[l][j] == m]) + alpha
                    posterior_probability[str(m) + str(n)] = numerator / denominator
            self.posterior[j] = posterior_probability

    def calculateProbability1(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateProbability2(self, x, prior, posterior, classValue):
        prior_prob = prior[classValue]
        posterior_prob = posterior[str(x) + str(classValue)]
        feature_log_prob_ = prior_prob * np.log(posterior_prob)
        return feature_log_prob_

    def calculateClassProbabilities(self, inputVector):
        probabilities = {}
        print self.summaries
        for classValue, classSummaries in self.summaries.iteritems():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                if i not in self.integerIndexList:
                    probabilities[classValue] *= self.calculateProbability1(x, mean, stdev)
                else:
                    probabilities[classValue] *= self.calculateProbability2(x, self.prior, self.posterior[i], classValue)
        return probabilities


    def predictLabel(self, inputVector):
        probabilities = self.calculateClassProbabilities(inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel


    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            result = self.predictLabel(X[i])
            predictions.append(result)
        return predictions


    def PredictionAccuracy(self, X, Y_test):
        predictions = self.predict(X)
        correct = 0
        for i in range(len(Y_test)):
            if Y_test[i] == predictions[i]:
                correct += 1
        return (correct / float(len(Y_test))) * 100.0


