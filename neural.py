import numpy
import math

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

    def train():
        pass

    def query():
        pass

def main():
    inputNodes = 3
    hiddenNodes = 3
    outputNodes = 3
    learningRate = 0.3

    n = neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    print(n.weightsInputToHidden)

if __name__ == "__main__":
    main()

