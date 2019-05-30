import tensorflow as tf
import numpy as np
print(tf.__version__)

s = tf.InteractiveSession()

tf.reset_default_graph()
# x = tf.get_variable("x", shape=(), dtype=tf.float32, trainable=True)
# x = tf.Variable(tf.ones((2, 2)))


# x = tf.get_variable("x", shape=(), dtype=tf.float32, trainable=True)
# x = tf.get_variable("x", initializer=[2.0])
x = tf.Variable(3.0)
f = x ** 2
f = tf.Print(f, [x, f], "x, f:")

optimizer = tf.train.GradientDescentOptimizer(0.2)
step = optimizer.minimize(f, var_list=[x])

print(tf.trainable_variables())

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    for i in range(10):
        s.run([step, f])

