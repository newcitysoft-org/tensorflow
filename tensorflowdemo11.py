# coding=utf-8
'''
    带命名空间，统计的手写数字训练模型可视化
'''
import input_data
import tensorflow as tf

# 获取数据训练集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples

# 参数摘要
def variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean',mean)#平均值
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
    tf.summary.scalar('stddev',stddev)
    tf.summary.scalar('max',tf.reduce_max(var))
    tf.summary.scalar('min',tf.reduce_min(var))
    tf.summary.histogram('histogram',var)

# 命名空间 定义placeholder
with tf.name_scope('input'):
    # 定义输入数组图片28*28=784长度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 定义输出数组种类10种
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]),name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# 交叉墒
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss',loss)
# 梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#转化结果值
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # argmax返回一维张量最大的值所在的位置
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求精确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 初始化变量
init = tf.global_variables_initializer()

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs', sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step], feed_dict={x: batch_xs, y: batch_ys})
            writer.add_summary(summary,batch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(" Iter " + str(epoch) + " ,Testing Accuracy " + str(acc))
