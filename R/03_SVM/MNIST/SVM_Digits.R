# R example
# SVM - MNIST Analysis
#
# Prof. Dr. Thorsten Schmidt, University Freiburg
# www.stochastik.uni-freiburg.de
# part of the AI lecture, SS 2017
# 

# Taken from https://gist.github.com/brendano/39760
# Copyright under the MIT license:
# I hereby license it as follows. This is the MIT license.
# Copyright 2008, Brendan O'Connor
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# Load the MNIST digit recognition dataset into R
# http://yann.lecun.com/exdb/mnist/
# assume you have all 4 files and gunzip'd them (already provided with this file)
# creates train$n, train$x, train$y  and test$n, test$x, test$y
# e.g. train$x is a 60000 x 784 matrix, each row is one digit (28x28)
# call:  show_digit(train$x[5,])   to see a digit.
# brendan o'connor - gist.github.com/39760 - anyall.org

library(e1071)

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


setwd("localpathtoMNIST")
load_mnist()

# We plot the first 100 images
par(mfrow=c(10,10),mai=c(0,0,0,0))
for (i in 1:100) {
	show_digit(train$x[i,],col=gray(12:1/12),xaxt="n", yaxt="n")
}

N=2000
dat=data.frame(x=train$x[1:N,],y=as.factor(train$y[1:N]))
svmfit = svm (y ~., data=dat,   method="class", kernel="linear", cost=10, scale=FALSE)

#prediction
testdat = data.frame(x=test$x[1:100,])
pred = predict(svmfit,testdat)

table(pred,test$y[1:100])

# The error rate 
sprintf("Error rate: %f",1-sum(pred == test$y[1:100])/100)

# Reaches an error rate of 4%. Can you do better ?

# Summarizing plot
par(mfrow=c(10,10),mai=c(0,0,0,0))
for (i in 1:100) {
	if (pred[i]==test$y[i]) {
		colt=gray(12:1/12)
	} else
	{
		colt=c(rgb(1,0.7,0.7),gray(11:1/11))
	}
	
	show_digit(test$x[i,],col=colt,xaxt="n", yaxt="n")
}



test$y[9]   # 5
pred[5]     # 4

test$y[55]   # 5
pred[55]     # 2