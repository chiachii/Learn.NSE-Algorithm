import numpy as np
import math
import copy as cp
from sklearn.ensemble import RandomForestClassifier

# define dot function: calculate the inner product used in recompute voting weight.
def dot(K, L):
   if len(K) != len(L):
      return 0
   return sum(i[0] * i[1] for i in zip(K, L))

class LearnNSE:
    '''Learn++.NSE ensemble classifier
    <Parameters>
    @base_classifier: arbitary supervised classifier (default=RandomForestClassifier)
    @class_num: int (default=10)
    @slope: float (default=0.5)
    @crossing_point: float (default=10.0)
    @models: list (default=None)
    @voting_weights: list (default=None)
    @error_weights: list (default=None)
    @error_distribution: list (default=None)
    '''
    def __init__(self, base_classifier=RandomForestClassifier(n_estimators=100, random_state=10),
                 alpha=0.5, beta=10.0):
        self.base_classifier = cp.deepcopy(base_classifier) # reset a model for current dataset
        self.models = []
        self.slope = alpha
        self.crossing_point = beta
        self.voting_weights = [1.0] # default=1.0 
        self.error_distribution = []
        self.bkts = [] # save beta computed from the formula based on punishment of error rate
        self.wkts = []

    def fit(self, X_train, y_train):
        '''Function fit(): training model, and ensemble them
        <Parameters>
        @X_train: Dataframe
            A multi-dimension dataset for training model
        @y_train: Dataframe {0,1,2,...,n}
            The set of label of each line of training data
        @base_classifier: arbitary supervised classifier (default=RandomForestClassifier)
            For training new sub-classifier into self.models
        '''
        clf = cp.deepcopy(self.base_classifier)
        clf.fit(X_train, y_train)
        self.models.append(clf)

    def predict(self, X_test):
        '''Function predict(): testing model, and get the result from the ensemble model
        <Parameters>
        @X_test: Dataframe
            A multi-dimension dataset for testing model
        
        <Returns>
        @y_pred: numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X
        '''
        y_pred = []
        t = len(self.models)
        for idx in range(len(X_test)):
            weighted_pred = [0.0]
            temp_pred = []
            for k in range(1, t+1):
                clf = self.models[k-1]
                temp_target = clf.predict(X_test[idx:idx+1]) # Take a column of data each time to predict
                # add new label
                if (temp_target[0]+1) > len(weighted_pred):
                    weighted_pred.append(0.0)
                # add voting weight in corresponding index
                weighted_pred[temp_target[0]] += self.voting_weights[k-1]

            temp_pred.append(weighted_pred.index(max(weighted_pred))) # Get the Max value in the List as prediction of that row (X)
            y_pred.append(temp_pred)
        return np.array(y_pred) # Todo: prediction needs to time the weight
    
    def score(self, X_test, y_test):
        '''Function score(): testing model, and ensemble them
        <Parameters>
        @X_test: Dataframe
            A multi-dimension dataset for testing model
        @y_test: Dataframe
            The set of label of each line of testing data
        @score_list: List
            Save the prediction accuracy in binary 
        '''
        score_list = []
        y_pred = self.predict(X_test)
        for idx in range(len(X_test)):
            if y_pred[idx] == y_test.values[idx][0]:
                score_list.append(1)
            else:
                score_list.append(0)
        return np.sum(score_list)/len(score_list)
    
    def redistribute_error_rate(self, X_train, y_train): # number of models = t-1
        '''Function redistribute_error_rate(): redistribution on newest dataset for evaluate all of the classifier in Learn++.NSE
        <Parameters>
        @X_train: Dataframe
            A multi-dimension dataset for training model
        @y_train: Dataframe {0,1,2,...,n}
            The set of label of each line of training data
        '''
        error_distribution = []
        ErrorRate = 1.0-self.score(X_train, y_train) 
        y_pred = self.predict(X_train)
        for idx in range(len(X_train)):
            if y_pred[idx] == y_train.values[idx][0]:
                error_distribution.append(ErrorRate)
            else:
                error_distribution.append(1)
        self.error_distribution = error_distribution

    def revoting(self, X_train, y_train): # number of models = t
        '''Function revoting(): update the weight based on error rate for output the prediction
        <Parameters>
        @X_train: Dataframe
            A multi-dimension dataset for training model
        @y_train: Dataframe {0,1,2,...,n}
            The set of label of each line of training data
        @ekt: float
            The punishment of each sub-classifier
        '''
        # check whether self.error_distribution need initialization
        if len(self.error_distribution) == 0:
            self.error_distribution = [1/len(X_train)]*len(X_train)

        ##### Step 5-1. Compute the Error-based Weight #####
        # bkt_list = [] # Record penalties until all models have been evaluated
        t = len(self.models)
        self.bkts.append([])  
        for k in range(1, t+1):
            clf = self.models[k-1]
            ekt = 0
            y_pred = clf.predict(X_train)
            for idx in range(len(X_train)):
                if y_pred[idx] != y_train.values[idx][0]:
                    ekt += self.error_distribution[idx]
            
            ekt = ekt/np.sum(self.error_distribution)
            # print('ekt={}'.format(ekt)) # check the performance of each model on newest dataset
            if ekt > 0.5:
                bkt = 0.5/(1-0.5)
            else:
                bkt = ekt/(1-ekt)
            # store normalized error for this classifier
            self.bkts[k-1].append(bkt)
        #####################################################

        ##### Step 5-2. Compute the Time-based Weigh t#####
        # compute the (time) weighte for each model on current time step
        curr_wkt_list = []
        self.wkts.append([])    
        for k in range(1, t+1):
            wkt = 1.0 / (1.0 + np.exp((-1)*self.slope*(t - k - self.crossing_point)))
            curr_wkt_list.append(wkt)
        # compute the (time) weighted normalized errors for kth classifier h_k
        t = len(self.models)
        for k in range(1, t+1):
            wkt = curr_wkt_list[k-1]
            
            if len(self.wkts[k-1]) != 0:
                wkt = wkt/(np.sum(self.wkts[k-1]) + wkt)
            else:
                wkt = wkt/wkt # the time weight of newest model 
            # store the normalized (time) errors
            self.wkts[k-1].append(wkt)
        ####################################################

        ##### Step 6. Calculate the voting weight #####
        voting_weight_list = []
        for k in range(1, t+1):
            TimeAndErrorWeight = np.sum(dot(self.bkts[k-1], self.wkts[k-1])) + 5e-2 # add deviation 5e-2 to avoid that the value equals to 0
            voting_weight_list.append(np.log(1/TimeAndErrorWeight))
        self.voting_weights = voting_weight_list
        ###############################################
