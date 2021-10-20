import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
"""import pandas as pd"""
from sklearn import datasets

sess = tf.compat.v1.Session()
iris = datasets.load_iris()

"""
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.values)
co = df.corr()
"""

x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([x[2] for x in iris.data])

plt.clf()
plt.plot(x_vals, y_vals, 'o', label = "Datos")
plt.savefig("img/RNN-5_data")

t1 = time.time()

#Modelo y = Ax + b
learning_rate = 0.05
batch_size = 25
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

y_predict = tf.add(tf.matmul(x_data,A),b)

loss_l2 = tf.reduce_mean(tf.square(y_target-y_predict))
my_optim = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = my_optim.minimize(loss_l2)
init = tf.global_variables_initializer()
sess.run(init)

loss_vect_l2 = []
for i in range(100):
    rand_idx = np.random.choice(len(x_vals), size = batch_size)
    rand_x = np.transpose([x_vals[rand_idx]])
    rand_y = np.transpose([y_vals[rand_idx]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss_l2, feed_dict={x_data:rand_x, y_target: rand_y})
    loss_vect_l2.append(temp_loss)
    if(i+1)%10==0:
        print("Paso #"+str(i+1)+", A="+str(sess.run(A))+", b="+str(sess.run(b))+
             ", Loss = "+str(temp_loss))

print(str(t1 - time.time()))

[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
print(str(slope)+"x+"+str(intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope*i+intercept)

plt.clf()
plt.plot(x_vals, y_vals, 'o', label = "Datos")
plt.plot(x_vals, best_fit, 'r-', label="Recta de regresión con TF", linewidth=3)
plt.legend(loc = "upper left")
plt.title("Longitud de sépalos vs Anchura de Pétalos")
plt.xlabel("Anchura del Pétalo")
plt.ylabel("Longitud del Sépalo")
plt.savefig("img/RNN-5_predict")

plt.clf()
plt.plot(loss_vect_l2, "k-")
plt.title("Función de pérdidas con L2")
plt.xlabel("Iteración del algoritmo")
plt.ylabel("Medida de L2")
plt.savefig("img/RNN-5_loss")

