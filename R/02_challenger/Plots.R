# R example
# Challenger data / logistic regression
#
# Prof. Dr. Thorsten Schmidt, University Freiburg
# www.stochastik.uni-freiburg.de
# part of the AI lecture, SS 2017


# Logistic function
x=seq(-6,6,by=0.01)
y = exp(x) / (1+exp(x))

plot(x,y,type="l",main="Logistic function",lwd=2)
lines(x,pnorm(x),lty=2,lwd=2)
legend(-5.6,0.944,lty=c(1,2),legend=c("logistic","normal"))


# Logit Function
x=seq(0,1,by=0.001)
y=log (x/(1-x))
plot(x,y,type="l",main="Logit function",lwd=2)


library(gdata,quietly=TRUE,verbose=FALSE, warn.conflicts=FALSE)  # for reading xls
setwd ("MakeSureToSetTheRightDirectoryHere")

data = read.xls("ChallengerData.xls") # Taken From Casella & Berger (2002)


plot (data$Temp, data$Failure,xlim=c(30,85))

# Perform the regression
summary(out.int <- glm(Failure ~ Temp, family=binomial , data = data))
a= out.int$coefficients[1]
b= out.int$coefficients[2]

# plot the result
x=seq(30,85,by=1)
lines(x,exp(a+b*x)/(1+exp(a+b*x)),lwd=2,lty=2)

# Confidence bounds
lines (c(x[23],x[23]),c(0,1),lty=2,lwd=2,col="gray")
lines (c(x[49],x[49]),c(0,1),lty=2,lwd=2,col="gray")

# What is the value for x=31?
x=31
exp(a+b*x)/(1+exp(a+b*x))