import random as RD
import math
from sklearn.model_selection import KFold
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd


class GRNN:
    def __init__(self, numgates, nummods, train_x, train_y, gates):
        self.num_gates = numgates  # number of gate variables
        self.num_models = nummods  # number of machine learning models
        self.num_trainings = len(train_x)  # size of training data
        self.sigmas = gates  # each gate has a sigma value
        self.training_x = train_x  # training data
        self.training_y = train_y
        self.accuracy = []
        self.accuracy_rate = 0.0
        cv = KFold(n_splits=4)
        self.model_results = [[0.0 for m in range(self.num_models)] for t in range(self.num_trainings)]
        for train_ix, test_ix in cv.split(self.training_x):
            self.crossValidationEachFold(train_ix, test_ix)
        self.accuracy_rate = np.mean(self.accuracy)

    def Predict(self, holdout, holdoutList, gateIndexList = False):
        GateIndexes = range(self.num_gates)
        if gateIndexList != False:
            GateIndexes = gateIndexList
        errors = range(self.num_models) # [error1, error2, error3] each error for each machine learning model
        weights = range(self.num_models) # [weight1, weight2, weight3] each weight for each machine learning models
        for k in range(self.num_models): # loop over all machine learning models
            num, den = 0, 0
            for i in holdoutList: # loop over all the training data row by ro
                dist = 0
                for j in GateIndexes: # loop over all variables! all variables are gate variables
                    temp = (self.training_x[holdout][j] - self.training_x[i][j]) / self.sigmas[j]
                    dist += temp * temp
                temp = float(self.training_y[i]) - float(self.model_results[i][k])
                num += temp * temp * math.exp(-dist) # num +=
                den += math.exp(-dist) #den +=
            errors[k] = num / den

        temp = 0.0
        for k in range(self.num_models):
            temp = temp + (1 / errors[k])

        for k in range(self.num_models):
            weights[k] = (1 / errors[k]) / temp

        temp = 0.0
        for k in range(self.num_models):
            temp = temp + float(self.model_results[holdout][k]) * weights[k]
        return temp



    def calcAccuracy(self, predict, actual):
        right, wrong = 0.0, 0.0
        for i in xrange(len(predict)):
            if float(actual[i]) == 0.0:
                if float(predict[i]) >=0.5:
                    right += 1.0
                else:
                    wrong += 1.0
            else:
                if float(predict[i]) < 0.5:
                    right += 1.0
                else:
                    wrong += 1.0
        print right, wrong
        return float(right/(right + wrong))


    def numberProtection(self, score):
        if score == 1.0:
            return 0.99
        elif score == 0.0:
            return 0.01
        else:
            return score


    def crossValidationEachFold(self, trainingIndex, testingIndex):
        self.predict = range(self.num_trainings)
        self.actual = range(self.num_trainings)
        clf1 = GaussianNB()
        clf2 = KNeighborsClassifier()
        clf3 = LogisticRegression()
        #print self.training_x[trainingIndex][0], self.training_y[trainingIndex][0]
        clf1.fit(self.training_x[trainingIndex], [i[0] for i in self.training_y[trainingIndex]])
        results1 = clf1.predict_proba(self.training_x[testingIndex])
        clf2.fit(self.training_x[trainingIndex], [i[0] for i  in self.training_y[trainingIndex]])
        results2 = clf2.predict_proba(self.training_x[testingIndex])
        clf3.fit(self.training_x[trainingIndex], [i[0] for i in self.training_y[trainingIndex]])
        results3 = clf3.predict_proba(self.training_x[testingIndex])
        for i in xrange(len(testingIndex)):
            self.model_results[testingIndex[i]][0] = self.numberProtection(results1[i][0])
            self.model_results[testingIndex[i]][1] = self.numberProtection(results2[i][0])
            self.model_results[testingIndex[i]][2] = self.numberProtection(results3[i][0])

        for z in testingIndex:
            self.predict[z] = self.Predict(z, holdoutList = testingIndex)
            self.actual[z] = self.training_y[z]
        self.cvAccuracy = self.calcAccuracy([self.predict[i] for i in testingIndex], [self.actual[i] for i in testingIndex])
        self.accuracy.append(self.cvAccuracy)




class DifferentialEvolution:
    def __init__(self, popsize, mutrate, crossrate, minsig, maxsig, gens):
        self.population = range(popsize)
        self.mut_rate = mutrate
        self.cross_rate = crossrate
        self.min_sig = minsig
        self.max_sig = maxsig
        self.num_gens = gens


    def loadData(self, numgates, nummods, train_x, train_y):
        self.num_gates = numgates
        self.num_models = nummods
        self.num_trainings = len(train_x)
        self.training_data = [train_x, train_y]

        print ("========  Data Successfully Loaded ===========")

    def createPop(self):
        for g in range(len(self.population)):
            togate = range(self.num_gates)
            for h in range(self.num_gates):
                togate[h] = (RD.random() * (self.max_sig - self.min_sig)) + self.min_sig
            self.population[g] = GRNN(self.num_gates, self.num_models, self.training_data[0], self.training_data[1],  togate)
            print ("Chromosome ", g, "Created with Fitness Value: ", self.population[g].accuracy_rate)
        print ("Population Created")

    def loadPop(self, loadname, sizer): # loadData(14, 3, 89, 'View1.txt')
        g = open(loadname)
        glines = g.readlines()
        self.population = range(sizer)
        togate = range(sizer)
        for i in range(sizer):
            togate[i] = glines[i].split(' ')
            for u in range(self.num_gates):
                togate[i][u] = float(togate[i][u])
            self.population[i] = GRNN(self.num_gates, self.num_models,  self.training_data[0], self.training_data[1], togate[i])

    def savePop(self, outfile):
        h = open(outfile, 'w')
        for p in range(len(self.population)):
            for x in range(self.num_gates):
                h.write(str(self.population[p].sigmas[x]))
                if x < self.num_gates - 1:
                    h.write(' ')
            h.write('\n')
        h.close()

    def mutate_cross_replace(self, parent):
        if len(self.population) < 4.0:
            print ("Differential Equation Population TOO SMALL")
            return
        DE_list = range(len(self.population)) # DElist only include the index of all nodes
        DE_list.pop(parent) # remote the parent node from DElist
        base, diff1, diff2 = RD.sample(DE_list, 3) # randomly select three nodes from DElist
        togate = range(self.num_gates) #initial a new list store togate values
        for r in range(self.num_gates):
            if RD.random() < self.cross_rate:
                togate[r] = self.population[parent].sigmas[r]
            else:
                togate[r] = self.population[base].sigmas[r] + (self.mut_rate * abs(self.population[diff1].sigmas[r] - self.population[diff2].sigmas[r]))
        child = GRNN(self.num_gates, self.num_models, self.training_data[0], self.training_data[1], togate)
        if child.accuracy_rate > self.population[parent].accuracy_rate:
            self.population[parent] = child
            print ("Chromosome ", parent, "replaced with new accuracy: ", self.population[parent].accuracy_rate)
        else:
            print ("Chromosome ", parent, "not replaced")

    def runDE(self):
        for t in range(self.num_gens):
            print ("============================")
            print ("{} : {}".format("This the the Generation", t))
            self.maxAccuracy = 0
            self.hasmax = 0
            for y in range(len(self.population)):
                self.mutate_cross_replace(y)
                if self.population[y].accuracy_rate > self.maxAccuracy:
                    self.maxAccuracy = self.population[y].accuracy_rate
                    self.hasmax = y
            print ("Generation: ", t, " ends")
            print ("Best in population is Chromosome ", self.hasmax, " with accuracy: ", self.maxAccuracy)
            print ("Best Accuracy: " + str(self.maxAccuracy))


class RunGRNN():
    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def run(self):
        abc = DifferentialEvolution(20, .4, .4, .5, 1., 20)
        abc.loadData(5, 3, self.train_x, self.train_y)
        abc.createPop()
        abc.runDE()
        abc.savePop('firstsave.txt')
