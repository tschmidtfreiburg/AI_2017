# R example
# MxNEt - MNIST Analysis
#
# Prof. Dr. Thorsten Schmidt, University Freiburg
# www.stochastik.uni-freiburg.de
# part of the AI lecture, SS 2017
# 

# In parts taken from https://gist.github.com/brendano/39760
# Copyright under the MIT license:
# I hereby license it as follows. This is the MIT license.
# Copyright 2008, Brendan O'Connor
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# In parts taken from http://tjo-en.hatenablog.com/entry/2016/03/30/233848
# See also the nice explnation on
# https://mxnet.incubator.apache.org/tutorials/python/mnist.html

# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them (already provided with this file)
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org


library(mxnet)


load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('train-images-idx3-ubyte')     # Images
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')   # Labels
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}


 setwd("localpathtoMNIST")   # Insert you path here
# eg.:
# setwd(".../AI_2017/R/2_SVM/MNIST")


load_mnist()

# lineraly transform it into [0,1]
# transpose input matrix to npixel x nexamples (format expected by mxnet)

train.x = data.matrix(train$x)/255
train.y = train$y

test.x = data.matrix(test$x)/255
test.y = test$y

# Data load & preprocessing


  

# We plot the first 100 images
par(mfrow=c(10,10),mai=c(0,0,0,0))
for (i in 1:100) {
	show_digit(train$x[i,],col=gray(12:1/12),xaxt="n", yaxt="n")
}



# Configure the structure of the network

#in mxnet use its own data type symbol to configure the network
data <- mx.symbol.Variable("data")

#set the first hidden layer where data is the input, name and number of hidden neurons
fc1 <- mx.symbol.FullyConnected(data, name = "fc1", num_hidden = 128)

#set the activation which takes the output from the first hidden layer fc1
act1 <- mx.symbol.Activation(fc1, name = "relu1", act_type = "relu")

#second hidden layer takes the result from act1 as the input, name and number of hidden neurons
fc2 <- mx.symbol.FullyConnected(act1, name = "fc2", num_hidden = 64)

#second activation which takes the output from the second hidden layer fc2
act2 <- mx.symbol.Activation(fc2, name = "relu2", act_type = "relu")

#this is the output layer where number of nuerons is set to 10 because there's only 10 digits
fc3 <- mx.symbol.FullyConnected(act2, name = "fc3", num_hidden = 10)

#finally set the activation to softmax to get a probabilistic prediction
softmax <- mx.symbol.SoftmaxOutput(fc3, name = "sm")




#set which device to use before we start the computation
#assign cpu to mxnet
devices <- mx.cpu()

#set seed to control the random process in mxnet
mx.set.seed(0)

#train the neural network
model <- mx.model.FeedForward.create(softmax, 
                                     X = train.x, 
                                     y = train.y,
                                     ctx = devices, 
                                     num.round = 10, 
                                     array.batch.size = 100,
                                     learning.rate = 0.07, 
                                     momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     initializer = mx.init.uniform(0.07),
                                     epoch.end.callback = mx.callback.log.train.metric(100))




#make a prediction
preds <- predict(model, test.x)


#matrix with 10000 rows and 10 columns containing the classification probabilities from the output layer
#use max.col to extract the maximum label for each row
pred_label <- max.col(t(preds)) - 1
table(pred_label)

# Compute accuracy
acc <- sum(test.y == pred_label)/length(test.y)
print(acc)
# 97.23 % 

# We plot some of the test images
par(mfrow=c(10,10),mai=c(0,0,0,0))
for (i in 901:1000) {
	if (test.y[i]!=pred_label[i]){
	show_digit(test$x[i,],col=gray(12:1/12),xaxt="n", yaxt="n") }
	else {
		show_digit(rep(0,784),col=gray(12:1/12),xaxt="n", yaxt="n")
	}
}


# Now we perform the convolutional network 


data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                            kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                            kernel=c(2,2), stride=c(2,2))
                            
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")

# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)

# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)


# Prepare input
train.array <- t(train.x)
dim(train.array) <- c(28, 28, 1, ncol(train.array))
test.array <- t(test.x)
dim(test.array) <- c(28, 28, 1, ncol(test.array))


mx.set.seed(0)
tic <- proc.time()


model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                      ctx=devices, num.round=10, array.batch.size=100,
                                      learning.rate=0.05, momentum=0.9, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      epoch.end.callback=mx.callback.log.train.metric(100))



#make a prediction
preds <- predict(model, test.array)


#matrix with 10000 rows and 10 columns containing the classification probabilities from the output layer
#use max.col to extract the maximum label for each row
pred_label <- max.col(t(preds)) - 1
table(pred_label)

# Compute accuracy
acc <- sum(test.y == pred_label)/length(test.y)
print(acc)
# 99.01% - Amazing ! 

# We plot some of the test images
par(mfrow=c(10,10),mai=c(0,0,0,0))
for (i in 901:1000) {
	if (test.y[i]!=pred_label[i]){
	show_digit(test$x[i,],col=gray(12:1/12),xaxt="n", yaxt="n") }
	else {
		show_digit(rep(0,784),col=gray(12:1/12),xaxt="n", yaxt="n")
	}
}


