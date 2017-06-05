"""
Data : MNIST

Techniques what I used are
  - Multi-layered NN
  - Xavier's weight initialization
  - Dropout
  - ReLU (instead of sigmoid)
  - Softmax
  - Cross entropy
  - Adam Optimization (instead of Gradient Descent)

Test accuracy : 0.9818
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# setting
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6. / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3. / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def get_fc_w(name, shape):
    return tf.get_variable(name, shape=shape, initializer=xavier_init(*shape))


dropout_rate = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 784])     # input
y = tf.placeholder(tf.float32, [None, 10])      # real answers while training

_w1 = get_fc_w("w1", [784, 256])
w1 = tf.nn.dropout(_w1, dropout_rate)
_w2 = get_fc_w("w2", [256, 256])
w2 = tf.nn.dropout(_w2, dropout_rate)
_w3 = get_fc_w("w3", [256, 256])
w3 = tf.nn.dropout(_w3, dropout_rate)
_w4 = get_fc_w("w4", [256, 256])
w4 = tf.nn.dropout(_w4, dropout_rate)
_w5 = get_fc_w("w5", [256, 256])
w5 = tf.nn.dropout(_w5, dropout_rate)
_w6 = get_fc_w("w6", [256, 256])
w6 = tf.nn.dropout(_w6, dropout_rate)
w7 = get_fc_w("w7", [256, 10])

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([256]))
b4 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([256]))
b6 = tf.Variable(tf.random_normal([256]))
b7 = tf.Variable(tf.random_normal([10]))

layer1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, w3), b3))
layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, w4), b4))
layer5 = tf.nn.relu(tf.add(tf.matmul(layer4, w5), b5))
layer6 = tf.nn.relu(tf.add(tf.matmul(layer5, w6), b6))
hypothesis = tf.add(tf.matmul(layer6, w7), b7)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], dropout_rate: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: batch[0], y: batch[1], dropout_rate: 1}) / total_batch

        if epoch % display_step == 0:
            print("step %d, cost %.9f" % (epoch, avg_cost))

    print("Optimization finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1}))
