import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# generate random points
x_data = np.linspace(0., 5., 1000)
y_data = x_data * 0.5 - 4 + np.random.normal(scale=0.1, size=x_data.shape)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# weight & bias
w = tf.Variable(tf.random_uniform([1], -1., 1.))
b = tf.Variable(tf.random_uniform([1], -1., 1.))

# hypothesis for linear regression
hypothesis = w * x + b

# least square
cost = tf.reduce_mean(tf.square(hypothesis - y))

# gradient descent method
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(train, feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run(w), sess.run(b))

    y_estimated = sess.run(w) * x_data + sess.run(b)
    plt.plot(x_data, y_data, 'k,')  # black dot
    plt.plot(x_data, y_estimated, 'r')  # red line
    plt.show()
