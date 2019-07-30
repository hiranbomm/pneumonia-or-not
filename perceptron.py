import numpy as np

NUM_CLASS = 2  # pneumonia or normal




# 0 = normal
# 1 = pneumonia

class MultiClassPerceptron(object):

    def __init__(self, feature_dim):
        # for multi-class perceptrons, we have a different w*f for each category.
        # we will find the w*f for all w's and then find the max of these w*f's
        # return which class gives the max result

        # weights[0] holds the bias element
        self.weights = np.zeros(feature_dim + 1)
        self.learning_rate = 0.01
        self.threshold = 10  # how many times we want to go through training sets

    def prediction(self, img):


        sum_w = np.dot(img, self.weights[1:]) + self.weights[0]

        activation = 0

        if sum_w <= 0:
            activation = 1

        return activation


    def train(self, norm_train, pneu_train):
        """
        training algo:
        start with all zero weights (done in init step)
        pick up training instances one by one
        classify with current weights y = argmax_c(w_c * f) -> done in prediction method
        if correct prediction, no change.  (if what you classified it as matches the training label)
        if class c gets misclassified as c': (if what you classified it as does not match the training label)
        update for c: w_c = w_c + n*f
        update for c': w_c' = w_c' - n*f
        dont change anything in other class weight vectors!
        """

        # first go through norm_train

        for iter in range(self.threshold):
            expected = 0
            for img in norm_train:
                pred_class = self.prediction(img)

                # w = w + learning_rate * (expected - predicted) * x
                self.weights[1:] += self.learning_rate * (expected - pred_class) * img
                self.weights[0] += self.learning_rate * (expected - pred_class)

            expected = 1
            for img in pneu_train:
                pred_class = self.prediction(img)

                # w = w + learning_rate * (expected - predicted) * x
                self.weights[1:] += self.learning_rate * (expected - pred_class) * img
                self.weights[0] += self.learning_rate * (expected - pred_class)

    # def test(self, norm_test, pneu_test):


