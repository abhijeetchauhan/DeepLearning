from numpy import exp, array, random, dot
import numpy as np


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
        self.synaptic_weights = []
        # We model a 3 layer feedforward network, with 3 input connections, 2 hidden connection and 1 output connection.
        # We assign random weights to a 3 x 2 matrix, with values in the range -1 to 1 for input to hidden layer
        # and mean 0.
        # And assign random weights from hidden to output layer too.
        self.synaptic_weights.append(2 * random.random((3, 2)) - 1)
        self.synaptic_weights.append(2 * random.random((2,1)) - 1)

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = []
            error.append(training_set_outputs - output[1])


            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            # Backpropagate to output layer 2x4 . 4x1 = 2x1
            adjustment = []
            adjustment.append(dot(output[0].T, error[0] * self.__sigmoid_derivative(output[1])))
            # Backpropagate to hidden layer 4x1 . 1x2 = 4x2
            error.append(dot(error[0],adjustment[0].T))
            adjustment.append(dot(training_set_inputs.T, error[1] * self.__sigmoid_derivative(output[0])))

            
            # Adjust the weights.
            self.synaptic_weights[0] += adjustment[1][0]
            self.synaptic_weights[1] += adjustment[0][0]

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network input to hidden layer
        output = []
        hidden = self.__sigmoid(dot(inputs,self.synaptic_weights[0]))
        output.append(hidden)
        # from hidden to output layer
        output.append(self.__sigmoid(dot(hidden,self.synaptic_weights[1])))

        return output


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    # Test the neural network with a new situation.
    print "Considering new situation [1, 0, 1] -> ?: "
    print neural_network.think(array([1, 0, 1]))