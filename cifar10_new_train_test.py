import tensorflow as tf
from tensorflow import flags
import os
import cifar10_new
from datetime import datetime
import numpy as np
import time

flags.DEFINE_string('data_path','datasets/',
                    """Path to the CIFAR-10 data directory.""")
flags.DEFINE_string('train_dir', 'train_log/',
                           """Directory where to write event logs """)
flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
FLAGS=flags.FLAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=500
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL=200

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):

  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image,label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])
def read_cifar10(filename_queue):

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  label_bytes = 1  
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  record_bytes = tf.decode_raw(value, tf.uint8)

  result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32) 

  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])


  return result
def distorted_inputs(data_path, batch_size):

  filenames = [os.path.join(data_path, 'data_batch_%d.bin' % i)
               for i in range(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_cifar10(filename_queue)  #返回一个类
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = 24
  width = 24

  distorted_image = tf.random_crop(reshaped_image, [height, width,3])

  distorted_image = tf.image.random_flip_left_right(distorted_image)

  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  float_image = tf.image.per_image_standardization(distorted_image)

  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def inputs(eval_data, data_dir, batch_size):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = 24
  width = 24

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)

def running_train():
  global_step = tf.Variable(0, trainable=False)
  images, labels=distorted_inputs(data_dir_path,batch_size=128)
  logits = cifar10_new.inference(images)
  loss = cifar10_new.loss(logits, labels)
  train_op = cifar10_new.train(loss, global_step)
  saver = tf.train.Saver(tf.global_variables())
  init = tf.global_variables_initializer()
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
  sess.run(init)
  tf.train.start_queue_runners(sess=sess)
  for step in range(FLAGS.max_steps):
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])
    duration = time.time() - start_time

    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
      num_examples_per_step = 128
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
      print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  cifar10_new.eval_once(saver,top_k_op)        
def evaluate():
  test_epoch=cifar10_new.test_epoch
  images, labels = inputs(True,data_dir_path,batch_size=128)
  logits = cifar10_new.inference(images)
  top_k_op = tf.nn.in_top_k(logits, labels, 1)
  saver = tf.train.Saver()
  while True:
    cifar10_new.eval_once(saver, top_k_op)
    test_epoch-=1
    if test_epoch==0:
      break

if __name__ == '__main__':
  data_dir_path = os.path.join(FLAGS.data_path, 'cifar-10-batches-bin')
#  running_train()
  evaluate()
