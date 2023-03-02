import numpy as np
import math

class LearnNSE:
    def __init__(self, base_classifier, class_num=10, alpha=0.1, beta=0.5):
        self.base_classifier = base_classifier
        self.alpha = alpha
        self.beta = beta
        self.models = []
        self.weights = []
        self.time_weight_hist = []
        self.class_num = class_num

    def fit(self, X, y):
        # Set parameters
        error_weight = []
        time_weight = []

        # Initialize weights
        dataset_length = len(X)
        current_weights = [1.0]*dataset_length
        self.weights = []
        
        # Checks that there is at least one model in the model set
        if len(self.models) == 0:
            # Step1: Compute error of the existing ensemble on new data
            false_pred_num = 0
            for idx in range(dataset_length):
                if self.predict(X[idx]) == y[idx]:
                    continue
                else:
                    false_pred_num += 1
            Et = false_pred_num/dataset_length # Error of the existing ensemble

            # Step2: Update and Normalize instance weights
            for idx in range(dataset_length):
                if self.predict(X[idx]) == y[idx]:
                    current_weights[idx] = Et
                else:
                    current_weights[idx] = 1

        # Step3: Call BaseClassifier with new data
        new_model = self.base_classifier.fit(X)
        self.models.append(new_model)

        # Step4: Evaluate all existing classifiers on new data
        for sub_classifier in self.models:
            temp_error = 0
            for idx in range(dataset_length):
                if sub_classifier.predict(X[idx]) == y[idx]:
                    temp_error += current_weights[idx]
                else:
                    continue
            # Set the maximum value of the error to 1/2
            if temp_error > 0.5:
                temp_error = 0.5
            # Normalized the value of error
            temp_error = temp_error/(1-temp_error)
            # Add to the error list(@error_weight)
            error_weight.append(temp_error)

        # Step5: Compute the weighted average of all normalized errors for k-th classifier
        for idx in range(len(self.models)):
            power_value = (-1)*(self.alpha)*(idx-self.beta)
            temp_time_weight = 1/(1 + np.exp(power_value))
            if len(self.time_weight_hist):
                temp_time_weight = temp_time_weight/sum(self.time_weight_hist[idx])
                self.time_weight_hist.append(temp_time_weight)
            else:
                time_weight.append(temp_time_weight)
                self.time_weight_hist.append(time_weight)

        # Step6: Calculate classifier voting weights
        for idx in range(len(self.models)):
            beta_bar = sum(np.multiply(time_weight, error_weight))# the weight average of all normalized error
            final_weight = math.log(1/beta_bar)
        # Obtain the weight of each model in final hypothesis     
        self.weights.append(final_weight)

    # Step7: Obtain the final hypothesis    
    def predict(self, X):
        output = []
        y_pred = [0]*self.class_num
        for data_idx in range(len(X)):
            for model_idx in range(len(self.models)):
                sub_classifier = self.models[model_idx]
                target = sub_classifier.predict(X[data_idx])
                y_pred[target] += self.weights[model_idx]
            # Finish one instance
            output.append(y_pred.index(max(y_pred)))

        return output
        
