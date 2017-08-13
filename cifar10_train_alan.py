import numpy as np
import tensorflow as tf
import logging

from cifar10_input_alan import read_dataset

logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# DATASET_DIR = r'E:\Dataset\cifar-10-Python'
# DATASET_DIR = r'D:\MasterFiles\MachineLearning\TensorFlow\TensorFlow-Examples\MyCode\cifar10-dataset'
# DATASET_DIR = '/home/alan/文档/Datasets/cifar-10-Python'
DATASET_DIR = r'D:\lyj\cifar10\cifar-10-Python'
N_FEATURES = 3072
N_CLASSES = 10
N_FC1 = 384
N_FC2 = 192
BATCH_SIZE = 128
TEST_BATCH_SIZE = 5000
TRAINING_EPOCHS = 50000
DISPLAY_STEP = 5000
SAVE_STEP = 1000  # 保存模型频率
BASEDIR = './cifar10_model_hp_search/'
# L2 正则项系数
BETA = 0.01

# 读取 CIFAR10 训练集和验证集
cifar10 = read_dataset(DATASET_DIR, onehot_encoding=True)
logging.info('TRAIN: {}\nEVAL: {}'.format(cifar10.train.images.shape, cifar10.eval.images.shape))


def conv_layer(inpt, k, s, channels_in, channels_out, name='CONV'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(
            [k, k, channels_in, channels_out], stddev=0.1), name='W')
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


def fc_layer(inpt, neurons_in, neurons_out, last=False, name='FC'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(
            [neurons_in, neurons_out]), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[neurons_out]), name='b')
        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        if last:
            act = tf.add(tf.matmul(inpt, W), b)
        else:
            act = tf.nn.relu(tf.add(tf.matmul(inpt, W), b))
        tf.summary.histogram('activations', act)
        return act


def cifar10_model(learning_rate, batch_size):
    tf.reset_default_graph()
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, N_FEATURES], name='x')
        x_image = tf.transpose(tf.reshape(x, [-1, 3, 32, 32]), perm=[0, 2, 3, 1])
        tf.summary.image('input', x_image, max_outputs=3)
        y = tf.placeholder(tf.float32, [None, N_CLASSES], name='labels')

        # 是否处于训练阶段
        phase = tf.placeholder(tf.bool, name='PHASE')

        conv1 = conv_layer(x_image, 5, 1, channels_in=3, channels_out=64)
        with tf.name_scope('BN'):
            norm1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=phase)
        pool1 = pool_layer(norm1, 3, 2, pool_type='mean')
        conv2 = conv_layer(pool1, 5, 1, channels_in=64, channels_out=64)
        with tf.name_scope('BN'):
            norm2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=phase)
        pool2 = pool_layer(norm2, 3, 2, pool_type='mean')

        flattend = tf.reshape(pool2, shape=[-1, 8 * 8 * 64])
        fc1 = fc_layer(flattend, neurons_in=8 * 8 * 64, neurons_out=N_FC1)
        with tf.name_scope('BN'):
            norm3 = tf.contrib.layers.batch_norm(fc1, center=True, scale=True, is_training=phase)
        logits = fc_layer(norm3, neurons_in=N_FC1, neurons_out=N_CLASSES, last=True)

        # 模型参数，包括 weight 和 bias
        trainable_vars = tf.trainable_variables()

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)) + \
                BETA * tf.add_n([tf.nn.l2_loss(v)
                                for v in trainable_vars if not 'b' in v.name])

            tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        merged_summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        sess.run(init)

        LOGDIR = BASEDIR + 'lr={:.0E},bs={}'.format(learning_rate, batch_size)
        summary_writer = tf.summary.FileWriter(LOGDIR, graph=sess.graph)
        eval_writer = tf.summary.FileWriter(LOGDIR + '/eval')

        for i in range(TRAINING_EPOCHS):
            batch_x, batch_y = cifar10.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, phase: 1})
            if i % DISPLAY_STEP == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                s, lss, acc , _ = sess.run([merged_summary, loss, accuracy, train_step], 
                                    feed_dict={x: batch_x, y: batch_y, phase: 1},
                                    options=run_options,
                                    run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, 'step{}'.format(i))
                summary_writer.add_summary(s, i)
                for batch in range(cifar10.eval.num_exzamples // TEST_BATCH_SIZE):
                    test_acc = []
                    batch_x, batch_y = cifar10.eval.next_batch(TEST_BATCH_SIZE)
                    test_acc.append(sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, phase: 0}))
                eval_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='eval_accuracy', simple_value=np.mean(test_acc))]), i)
                logging.info('Iter={}, loss={}, trainging_accuracy={}, test_accuracy={}'.format(i+1, lss, acc, np.mean(test_acc)))
        LOGDIR = saver.save(sess, LOGDIR + '/model.ckpt')
        logging.info('Model saved in file: {}'.format(LOGDIR))


def main():
    for lr in [1e-2, 1e-3, 1e-4]:
        for bs in [64, 128]:
            logging.info('learing rate = {:.0E}, batch size = {}'.format(lr, bs))
            cifar10_model(lr, bs)
            
            # total_batches = cifar10.train.num_exzamples // BATCH_SIZE
            # logging.info('TOTAL BATCHES: {}'.format(total_batches))
            # for i in range(TRAINING_EPOCHS):
            #     avg_loss = 0
            #     for batch in range(total_batches):
            #         logging.debug('Iter {}, batch {}'.format(i, batch))
            #         batch_x, batch_y = cifar10.train.next_batch(BATCH_SIZE)
            #         _, l, acc = sess.run([train_step, loss, accuracy], feed_dict={x: batch_x, y: batch_y, phase: 1})
            #         avg_loss += l
            #         if i % DISPLAY_STEP == 0:
            #             s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y, phase: 1})
            #             summary_writer.add_summary(s, i)
            #     test_acc = sess.run(accuracy, feed_dict={x: cifar10.eval.images, y:cifar10.eval.labels, phase: 0})
            #     logging.info('Iter={}, loss={}, trainging_accuracy={}, test_accuracy={}'.format(i+1, avg_loss, acc, test_acc))
            
            
if __name__ == '__main__':
    main()