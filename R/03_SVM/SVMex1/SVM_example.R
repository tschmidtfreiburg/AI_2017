# R example
# SVM - Example
#
# Prof. Dr. Thorsten Schmidt, University Freiburg
# www.stochastik.uni-freiburg.de
# part of the AI lecture, SS 2017
# 

# Taken from  Chapter 9.6: Gareth James u. a. (2014). An Introduction to Statistical Learning: With Applications in R.

library(e1071)

#setseed(1)

N=20
x=matrix(rnorm(N*2), ncol=2)   # data - x corresponds to 2-dimensional data
y=c(rep(-1,N/2), rep(1,N/2))      # data - y has the classifier
x[y==1,]=x[y==1,] + 1           # We shift the mean for the second class by (1,1)
plot(x, col=(3-y))

dat = data.frame(x=x, y=as.factor(y))
svmfit = svm( y ~ ., data=dat, kernel="linear", cost=10, scale=FALSE)

plot(svmfit,dat,color = terrain.colors)

svmfit$index   # Identities of the support vectors
summary(svmfit)

# A smaller cost parameter

svmfit=svm(y ~., data=dat, kernel="linear", cost=0.1, scale=FALSE)
plot(svmfit , dat, color = terrain.colors)
svmfit$index

# Lets compare the cost vectors via the method tune

tune.out=tune(svm,y ~.,data=dat,kernel="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))
summary(tune.out)

#Parameter tuning of ‘svm’:
#- sampling method: 10-fold cross validation 
#- best parameters:
# cost
# 0.1
#- Detailed performance results:
#   cost error dispersion
#1 1e-03  0.65  0.3374743
#2 1e-02  0.65  0.3374743
#3 1e-01  0.05  0.1581139
#4 1e+00  0.10  0.2108185
#5 5e+00  0.15  0.2415229
#6 1e+01  0.15  0.2415229
#7 1e+02  0.15  0.2415229