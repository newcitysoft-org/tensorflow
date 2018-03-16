# coding=utf-8
'''

    没有优化的简单手写数字训练模型

'''
import input_data
import tensorflow as tf

# 获取数据训练集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples

# 定义输入数组图片28*28=784长度
x = tf.placeholder(tf.float32, [None, 784])
# 定义输出数组种类10种
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
# argmax返回一维张量最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求精确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(" Iter " + str(epoch) + " ,Testing Accuracy " + str(acc))
