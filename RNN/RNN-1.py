import tensorflow as tf
import numpy as np

sess = tf.compat.v1.Session()
my_array = np.array([
    [float(i) for i in range(2,11,2)],
    [float(i) for i in range(1,10,2)],
    [i*1.5 for i in range(1,10,2)]
])
print(my_array)

x_vals = np.array([my_array, my_array/2])
print(x_vals)

x_data = tf.placeholder(tf.float32, shape=(3,5))

m1 = tf.constant([[1.], [0.], [.5], [5.], [3.2]])
m2 = tf.constant([[7.]])
a1 = tf.constant([[2.]])

# (x*m1)*m2 + a1
prod1 = tf.matmul(x_data, m1)
prod2 = tf.matmul(prod1, m2)
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))

