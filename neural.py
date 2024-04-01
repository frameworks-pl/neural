import numpy
import math
import scipy.special #sigmoid function is in that lib (called expit())

class neuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        self.learningRate = learningRate

        #sampling weights from normal probability distribution centered around zero
        #and with a standard deviation that is related to number of incoming links into a node
        #1/sqrt(number of incoming links)
        self.weightsInputToHidden = numpy.random.normal(0.0, 1/math.sqrt(self.inputNodes), (self.hiddenNodes, self.inputNodes))
        self.weightsHiddenToOutput = numpy.random.normal(0.0, 1/math.sqrt(self.hiddenNodes), (self.outputNodes, self.hiddenNodes))

        #activation function using sigmoid function inside (sigmoid == scipy.special.expit)
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, targets_list):
        #convert inputs list to 2d array (with transpitions to 1,3 to 3,1)
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.weightsInputToHidden, inputs)
        #calculate the signals emerging from from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate singals into final output layer
        final_outputs = numpy.dot(self.weightsHiddenToOutput, hidden_outputs)

        #calculating errors
        output_errors = targets - final_outputs

        #propagating error (via weights) to hidden layer
        hidden_errors = numpy.dot(self.weightsHiddenToOutput.T, output_errors)

        #updating weights between hidden and output
        self.weightsHiddenToOutput += self.learningRate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        #updating weights between input and hidden layer
        self.weightsInputToHidden += self.learningRate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def query(self, input_list):
        #convert inputs list to 2d array (T gives transposition from 1,3, to 3,1)
        inputs = numpy.array(input_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.weightsInputToHidden, inputs)
        #calculate signals emerging from hiden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals info final output layer
        final_inputs = numpy.dot(self.weightsHiddenToOutput, hidden_outputs)
        #calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        

def main():
    inputNodes = 3
    hiddenNodes = 3
    outputNodes = 3
    learningRate = 0.3

    n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    #print(n.weightsInputToHidden)
    result = n.query([1.0, 0.5, -1.5])
    print(result)



if __name__ == "__main__":
    main()

