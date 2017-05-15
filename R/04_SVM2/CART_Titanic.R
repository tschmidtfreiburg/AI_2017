# R example
# CART
#
# Prof. Dr. Thorsten Schmidt, University Freiburg
# www.stochastik.uni-freiburg.de
# part of the AI lecture, SS 2017
# 

# Load library
library(rpart)
library(rpart.plot)

data(ptitanic)


# The plot is inspired by the Graphic from Stephen Milborrow  
# \url{https://commons.wikimedia.org/wiki/File:CART_tree_titanic_survivors.png} 

fit = rpart(survived ~  sex + age + sibsp, data=ptitanic, method="class")
cols <- c("darkred", "green4")[fit$frame$yval]
prp(fit,tweak=1.4 , extra=106, under=TRUE, ge=" > ", eq=" ", col=cols)

# Compare to an evt tree 
fite = evtree(survived ~  sex + age + sibsp  , data = ptitanic)
plot(fite)


# compare to the fine tuning from Trevor Stephens at his blog
# http://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/
# Note that he works with a different dataset from Kaggle  !

