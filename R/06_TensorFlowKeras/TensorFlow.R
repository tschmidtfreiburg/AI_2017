
# Taken from
# https://github.com/rstudio/tensorflow
#
# See also https://tensorflow.rstudio.com/tensorflow/articles/installation.html



install.packages("tensorflow")


library(tensorflow)
install_tensorflow()   # does all the work :)


cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y_conv), reduction_indices=1L))
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(y_conv, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(tf$global_variables_initializer())