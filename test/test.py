import numpy
import matplotlib.pyplot


data_file = open("mnist_train_100.csv", 'r')  
data_list = data_file.readline()
data_file.close()

all_values = data_list.split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap='Greys',  interpolation='None') 
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

print(scaled_input)
