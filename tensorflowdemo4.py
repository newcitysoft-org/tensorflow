# coding=utf-8
import input_data
import tensorflow as tf

# 获取数据训练集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(mnist.train.images)
# print(mnist.test.labels)

x = tf.placeholder(tf.float32, [None, 784])
# W:权重值
W = tf.Variable(tf.zeros([784, 10]))
# b:偏置量
b = tf.Variable(tf.zeros([10]))
# 核心:回归函数（原理：类似多维矩阵乘法）
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 输入正确值
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 梯度下降算法 最小化成本值(以0.01的学习速率最小化交叉熵)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 训练模型
# 初始化
init = tf.initialize_all_variables()
# 自动关闭session功能
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # print(i)
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy
    print ("评估值："+str(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))