
# ArtificialNeuralNetworks


Introduction

The objective of this work is to solve the MNIST Fashion classification problem using the
proposed neural network. The problem aims to classify input data (represented as a 784
element feature vector) with output data (a 10-element vector) which contains the learned
likelihood of the input’s classification at a given index. The MNIST Fashion dataset can be
classified into 10 different clothing items: t-shirt, trouser, pullover, dress, coat, sandal, shirt,
sneaker, bag, ankle boot. In this report, we focus on four networks we created to solve the
problem and discuss the tradeoffs offered by each. We pay special attention to how these so
called “prototypes” optimize performance with respect to accuracy and loss. We attempted to
find out what elements of the neural network, when modified, provided better results for
accurately guessing the inputted fashion images.
This relates to the class in that we have been learning about ways to train a neural network. We
started with basic algorithms, such as perceptrons, then went on to backpropagation, and now
finally we investigated convolutional neural networks. Convolutional neural networks are a
good fit for image classification because image classification involves so many different
variables that need to be accounted for. These variables can be accounted for by the many
hidden layers and their respective neurons using a convolutional architecture. We investigate
different architectures and make modifications to them based on what we have learned in
class.


