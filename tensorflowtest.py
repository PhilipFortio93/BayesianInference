import tensorflow as tf
import numpy as np
print(tf.__version__)


# generate model data
N = 1000
D = 3
x = np.random.random((N, D))
w = np.random.random((D, 1))
y = x @ w + np.random.randn(N, 1) * 0.20

print(x.shape, y.shape)
print(w.T)

tf.reset_default_graph()

features = tf.placeholder(tf.float32, shape=(None, D))
target = tf.placeholder(tf.float32, shape=(None, 1))

weights = tf.get_variable("weights", shape=(D, 1), dtype=tf.float32)
predictions = features @ weights

loss = tf.reduce_mean((target - predictions) ** 2)

print(target.shape, predictions.shape, loss.shape)

optimizer = tf.train.GradientDescentOptimizer(0.1)
step = optimizer.minimize(loss)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    for i in range(300):
        _, curr_loss, curr_weights = s.run([step, loss, weights], 
                                           feed_dict={features: x, target: y})
        if i % 50 == 0:
            print(curr_loss)

           