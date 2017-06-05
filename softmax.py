from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# setting
learning_rate = 0.1
training_epochs = 10
batch_size = 200
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])     # input
y = tf.placeholder(tf.float32, [None, 10])      # real answers while training

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.add(tf.matmul(x, w), b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
            avg_cost += sess.run(cost, feed_dict={x: batch[0], y: batch[1]}) / total_batch

        if epoch % display_step == 0:
            print("step %d, cost %.9f" % (epoch, avg_cost))

    print("Optimization finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
