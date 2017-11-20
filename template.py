import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_function = None,):
    Weights = tf.Variable(tf.random_normal(in_size,out_size))
    biase = tf.Variable(tf.zeros([1,out_size])+0.1,)
    Wx_plus_b = tf.matmul(inputs,Weights) + biase

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

#神经网络的input

#输出层

#预测值和真实值的误差


sess = tf.Session()

sess.run(tf.initialize_all_variables)

for i in range(1000):
    if i%50 == 0:
        print(i)






