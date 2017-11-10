import tensorflow as tf
import numpy as np

x_data = np.random.rand(1000).astype(np.float32)
y_data = x_data*0.1 + 0.3



Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimazier = tf.train.GradientDescentOptimizer(0.1)
train = optimazier.minimize(loss)

init = tf.global_variables_initializer()




sess = tf.Session()
sess.run(init)



for i in range(100):
    sess.run(train)
    print(i,sess.run(Weights),sess.run(biases))