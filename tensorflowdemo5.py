# coding=utf-8
import input_data
import tensorflow as tf


# 获取数据训练集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(mnist.train.images)

# 权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# x:输入正确值
x = tf.placeholder(tf.float32, [None, 784])

# W:权重值
W = tf.Variable(tf.zeros([784, 10]))

# b:偏置量
b = tf.Variable(tf.zeros([10]))

# 核心:回归函数（原理：类似多维矩阵乘法）
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 输入正确值
y_ = tf.placeholder("float", [None, 10])


# x转化为4d向量，2、3维对应图片的宽、高，4维对应图片颜色通道数(1:灰色，3:rgb彩色图)
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# ReLU神经元激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# max pooling处理结果
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# ReLU神经元激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# max pooling处理结果
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# 张量reshape成向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# ReLU神经元激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
# 初始化
sess = tf.InteractiveSession()
# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# ADAM优化器梯度最速下降算法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# boolean 数据集
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 转换为 0或1形式数据集
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())

for i in range(1000):
  batch = mnist.train.next_batch(100)
  if i % 100 == 0:
    # 比对
    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g" % (i, train_accuracy)
    # keep_prob来控制dropout比例
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 评估模型
print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.00})
