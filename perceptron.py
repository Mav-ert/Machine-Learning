#An implementation of perceptron based on description in Artificial Intelligence, A Guide For Thinking Humans by Melanie Mitchell

from warnings import simplefilter
import numpy as np
import random

class Perceptron(object):
    def __init__(self,no_input, learning_rate):
        self.threshold = random.random()
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-1,1,no_input)
        
    def show(self):
        print("Current Learning rate: ",  self.learning_rate)
        print("Threshold: ", self.threshold)
        print("Weights: ", self.weights)

    def predict(self, input):
        sum = np.dot(self.weights,input)+(-1*self.threshold)
        if sum > 0:          
            return 1      
        else:          
            return 0

    def train(self,training_inputs,labels,epochs):
        print("\ntraining for ", epochs, " rounds\n")
        for i in range(epochs):
            for x, t in zip(training_inputs,labels):
                for j in range(len(x)):
                    self.weights[j] += self.learning_rate * (t - self.predict(x)) * x[j]



#testing for and funciton
And = Perceptron(2,0.01)
And.show()

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([1, 0, 0, 0])

And.train(training_inputs, labels,180)
And.show()
print("predictions for And after training")
test_inputs = training_inputs
for i in test_inputs:
    print(i," : ",And.predict(i)) 


#testing for or
Or = Perceptron(2,0.01)
Or.show()

labels = np.array([1, 1, 1, 0])

Or.train(training_inputs, labels,180)
Or.show()
print("predictions for Or after training")
for i in test_inputs:
    print(i," : ",Or.predict(i)) 



#Testing for Xor
Xor = Perceptron(2,0.01)
Xor.show()

labels = np.array([0, 1, 1, 0])

Xor.train(training_inputs, labels,180)
Xor.show()
print("predictions for Xor after training")
for i in test_inputs:
    print(i," : ",Xor.predict(i)) 

# The perceptron cannot implement Xor gate why?
# perceptron with 2 inputs has x*w1 + y*w2
# t is threshold
# 1*w1 + 0*w2 >= t
# 0*w1 + 1*w2 >= t
# 0*w1 + 0*w2 < t
# 1*w1 + 1*w2 < t
# w1 >= t
# w2 >= t
# 0 < t
# w1+w2 < t 
# Contradiction occurs since w1 and w2 are >t then w1 + w2 cannot be <t

