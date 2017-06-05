"""
Data : MNIST

Techniques what I used are
  - Convolutional layer
  - Max-pooling layer
  - Fully-connected layer
  - Xavier's weight initialization
  - Dropout
  - ReLU (instead of sigmoid)
  - Softmax
  - Cross entropy
  - Adam Optimization

Test Accuracy : 0.9947
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6. / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3. / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def get_fc_w(name, shape):
    return tf.get_variable(name, shape=shape, initializer=xavier_init(*shape))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# setting
learning_rate = 0.001
training_epochs = 20
batch_size = 200
display_step = 1


conv_dropout_rate = tf.placeholder(tf.float32)
fc_dropout_rate = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

conv_w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1))
conv_w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1))
conv_w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1))

fc_w4 = get_fc_w('fc_w4', [4 * 4 * 128, 625])
fc_b4 = tf.Variable(tf.random_normal([625]))
fc_w5 = get_fc_w('fc_w5', [625, 10])
fc_b5 = tf.Variable(tf.random_normal([10]))

# [28, 28, 1] => [14, 14, 32]
conv_layer1 = tf.nn.relu(tf.nn.conv2d(x, conv_w1, strides=[1, 1, 1, 1], padding='SAME'))
pool_layer2 = tf.nn.max_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pool_layer2 = tf.nn.dropout(pool_layer2, conv_dropout_rate)

# [14, 14, 32] => [7, 7, 64]
conv_layer3 = tf.nn.relu(tf.nn.conv2d(pool_layer2, conv_w2, strides=[1, 1, 1, 1], padding='SAME'))
pool_layer4 = tf.nn.max_pool(conv_layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pool_layer4 = tf.nn.dropout(pool_layer4, conv_dropout_rate)

# [7, 7, 64] => [4, 4, 128]
conv_layer5 = tf.nn.relu(tf.nn.conv2d(pool_layer4, conv_w3, strides=[1, 1, 1, 1], padding='SAME'))
pool_layer6 = tf.nn.max_pool(conv_layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pool_layer6 = tf.nn.dropout(pool_layer6, conv_dropout_rate)

# [4, 4, 128] => [4 * 4 * 128]
pool_layer6 = tf.reshape(pool_layer6, [-1, 4 * 4 * 128])

# [4*4*128] => [625]
fc_layer7 = tf.nn.relu(tf.add(tf.matmul(pool_layer6, fc_w4), fc_b4))

# [625] => [10]
fc_layer7 = tf.nn.dropout(fc_layer7, fc_dropout_rate)
hypothesis = tf.add(tf.matmul(fc_layer7, fc_w5), fc_b5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            images = batch[0].reshape(-1, 28, 28, 1)
            labels = batch[1]
            sess.run(optimizer, feed_dict={x: images, y: labels, conv_dropout_rate: 0.8, fc_dropout_rate: 0.5})
            avg_cost += cost.eval({x: images, y: labels, conv_dropout_rate: 1, fc_dropout_rate: 1}) / total_batch

        if epoch % display_step == 0:
            print("step %d, cost %.9f" % (epoch, avg_cost))

    print("Optimization finished!")

    test_x = mnist.test.images.reshape(-1, 28, 28, 1)
    test_y = mnist.test.labels

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    size = 100
    total_accuracy = 0

    # because of the lack of gpu memory
    for i in range(int(test_x.shape[0] / size)):
        index_start = i * size
        index_end = i * size + size

        images = test_x[index_start:index_end, :, :, :]
        labels = test_y[index_start:index_end]

        current_accuracy = accuracy.eval({x: images, y: labels, conv_dropout_rate: 1, fc_dropout_rate: 1})
        print("%d. Current %d Accuracy: %g" % (i+1, size, current_accuracy))
        total_accuracy = (total_accuracy * i + current_accuracy) / (i + 1)

    print("Total Accuracy : %f" % total_accuracy)
