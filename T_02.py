import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp


def add_layer(inputs,in_size,out_size,n_layer,activation_funciont=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]))
            #tf.histogram_summary(layer_name+'/weights',Weights)
            #修改成下面
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_funciont is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_funciont(Wx_plus_b)

        tf.summary.histogram(layer_name + '/outputs', outputs);
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')


l1 = add_layer(xs,1,10,n_layer=1,activation_funciont=tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer=2,activation_funciont=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction,name='square'),reduction_indices=[1],name='reduce_sum'),name='reduce_mean')
    tf.summary.scalar('loss',loss)

#小于1的值

with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()

merged = tf.summary.merge_all()

#2016-11月已删除此方法
#write = tf.train.SummaryWriter("logs/",sess.graph)
#write = tf.train.SummarySaverHook("logs/",sess.graph)
write = tf.summary.FileWriter("logs/",sess.graph)
#在控制台用这个命令查看（在程序根目录下）    tensorboard --logdir="logs"

sess.run(init)

fig = mp.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
mp.ion()#让绘图不会暂停整个程序
mp.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        mp.pause(0.1)

        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        write.add_summary(result,i)















