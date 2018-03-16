# coding=utf-8

# coding=utf-8
'''

    多层神经元网络，过拟合数字训练模型

'''
import tensorflow as tf

import input_data

# 获取数据训练集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 加载图片每个批次大小
batch_size = 100
# 批次总数
n_batch = mnist.train.num_examples

# 定义输入数组图片28*28=784长度
x = tf.placeholder(tf.float32, [None, 784])
# 定义输出数组种类10种
y = tf.placeholder(tf.float32, [None, 10])
# 1：神经元网络全部工作，0.7:70%神经元工作
keep_prob = tf.placeholder(tf.float32)

# 创建一个简单的神经网络
W1 = tf.Variable(tf.zeros([784, 2000]))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.zeros([2000, 2000]))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.zeros([2000, 1000]))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.zeros([1000, 10]))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 平均值：二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))
# 平均值：交叉墒
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
# argmax返回一维张量最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求平均值：精确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1})
        print(" Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + " Train Accuracy " + str(train_acc))
