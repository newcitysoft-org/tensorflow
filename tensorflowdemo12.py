# coding=utf-8
'''
    带命名空间，统计,三维分类图片的手写数字训练模型可视化
'''
import input_data
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# 获取数据训练集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 运行次数
max_step = 1001
# 图片数量
image_num = 3000
# 文件路径(默认当前）
DIR = ""

sess = tf.Session()
# 载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')



batch_size = 100

n_batch = mnist.train.num_examples


# 参数概要
def variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)  # 平均值
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)  # 标准差
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)  # 直方图


# 命名空间 定义placeholder
with tf.name_scope('input'):
    # 定义输入数组图片28*28=784长度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 定义输出数组种类10种
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 显示图片
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 交叉墒代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    # 梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
sess.run(tf.global_variables_initializer())

# 转化结果值
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # argmax返回一维张量最大的值所在的位置（布尔值）
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求精确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件
if tf.gfile.Exists(DIR + 'projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'projector/metadata.tsv')
    # tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
with open(DIR + 'projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# 合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'metadata.tsv'
embed.sprite.image_path = DIR + 'data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer,config)

for i in range(max_step):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys},options=run_options,run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata,'step%03d'%i)
    projector_writer.add_summary(summary, i)

    if i%100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(" Iter " + str(i) + " ,Testing Accuracy " + str(acc))


saver.save(sess,DIR + 'projector/a_model.ckpt',global_step=max_step)
projector_writer.close()
sess.close()
