import numpy as np
import tensorflow as tf

from cifar10_input_alan import read_dataset

DATASET_DIR = r'E:\Dataset\cifar-10-Python'
# DATASET_DIR = r'D:\MasterFiles\MachineLearning\TensorFlow\TensorFlow-Examples\MyCode\cifar10-dataset'
LEARNING_RATE = 1e-3
N_FEATURES = 3072
N_CLASSES = 10
N_FC1 = 64
N_FC2 = 128
BATCH_SIZE = 64
TRAINING_EPOCHS = 5000
DISPLAY_STEP = 10

# 读取 CIFAR10 训练集和验证集
cifar10 = read_dataset(DATASET_DIR, onehot_encoding=True)

def conv_layer(inpt, k, s, channels_in, channels_out, name='CONV'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([k, k, channels_in, channels_out], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='b')
        conv = tf.nn.conv2d(inpt, W, strides=[1, s, s, 1], padding='SAME')
        act = tf.nn.relu(conv)
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act


def pool_layer(inpt, k, s, pool_type='mean'):
    if pool_layer is 'mean':
        return tf.nn.avg_pool(inpt,
                              ksize=[1, k, k, 1],
                              strides=[1, s, s, 1],
                              padding='SAME',
                              name='POOL')
    else:
        return tf.nn.max_pool(inpt,
                              ksize=[1, k, k, 1],
                              strides=[1, s, s, 1],
                              padding='SAME',
                              name='POOL')



def fc_layer(inpt, neurons_in, neurons_out, name='FC'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([neurons_in, neurons_out]), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[neurons_out]), name='b')
        act = tf.nn.relu(tf.add(tf.matmul(inpt, W), b))
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        return act


x = tf.placeholder(tf.float32, shape=[None, N_FEATURES], name='x')
x_image = tf.reshape(x, [-1, 32, 32, 3])
tf.summary.image('input', x_image, max_outputs=BATCH_SIZE)
y = tf.placeholder(tf.float32, [None, N_CLASSES], name='labels')

conv1 = conv_layer(x_image, 5, 1, channels_in=3, channels_out=32)
pool1 = pool_layer(conv1, 3, 2, pool_type='mean')
conv2 = conv_layer(pool1, 5, 1, channels_in=32, channels_out=64)
pool2 = pool_layer(conv2, 3, 2, pool_type='mean')

flattend = tf.reshape(pool2, shape=[-1, 8*8*64])
fc1 = fc_layer(flattend, neurons_in=8*8*64, neurons_out=N_FC1)
logits = fc_layer(fc1, neurons_in=N_FC1, neurons_out=N_CLASSES)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('./cifar10_model/6', graph=tf.get_default_graph())
    for i in range(TRAINING_EPOCHS):
        batch_x, batch_y = cifar10.train.next_batch(BATCH_SIZE)
        if i % DISPLAY_STEP == 0:
            s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(s, i)
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
