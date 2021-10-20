from typing import List
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from numpy.core.fromnumeric import shape
import tensorflow as tf
from sklearn import datasets

#Variables
sess = tf.compat.v1.Session()
seed = 1
batch_size = 50
hidden_layer_nodes = 5
learning_rate = 0.005

#seed
tf.compat.v1.set_random_seed(seed)
np.random.seed(seed)

#Carga de datos
iris = datasets.load_iris()

x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])

train_idx = np.random.choice(len(x_vals), size=round(len(x_vals)*0.8), replace=False)
test_idx = np.array(list(set(range(len(x_vals)))-set(train_idx)))

x_vals_train = x_vals[train_idx]
y_vals_train = y_vals[train_idx]
x_vals_test = x_vals[test_idx]
y_vals_test = y_vals[test_idx]

def normalize_col(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

x_vals_train = np.nan_to_num(normalize_col(x_vals_train))
x_vals_test = np.nan_to_num(normalize_col(x_vals_test))



#Modelo RELU
x_data = tf.compat.v1.placeholder(shape=[None,3], dtype=tf.float32)
y_target = tf.compat.v1.placeholder(shape=[None,1], dtype=tf.float32)

A1 = tf.Variable(tf.random.normal(shape=[3,hidden_layer_nodes]))
b1 = tf.Variable(tf.random.normal(shape=[hidden_layer_nodes]))

A2 = tf.Variable(tf.random.normal(shape=[hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random.normal(shape=[1]))

hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data,A1),b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output,A2),b2))

loss = tf.reduce_mean(tf.square(final_output-y_target))
my_optim = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = my_optim.minimize(loss)


#Inicializar variables
init = tf.compat.v1.global_variables_initializer()
sess.run(init)


loss_vect = []
test_loss = []
for i in range(500):
    rand_idx = np.random.choice(len(x_vals_train), size = batch_size)
    rand_x = x_vals_train[rand_idx]
    rand_y = np.transpose([y_vals_train[rand_idx]])

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target: rand_y})
    loss_vect.append(temp_loss)

    temp_loss_test = sess.run(loss, feed_dict={x_data:x_vals_test, y_target: np.transpose([y_vals_test])})
    test_loss.append(temp_loss_test)

    if(i+1)%50==0:
        print("Paso #"+str(i+1)+", Loss = "+str(temp_loss))

plt.clf()
plt.plot(loss_vect, "r-", label="Pérdida Entrenamiento")
plt.plot(test_loss, "b--", label ="Pérdida Test")
plt.title("Pérdida (RMSE) per iteración")
plt.xlabel("Iteración")
plt.ylabel("RMSE")
plt.legend(loc ="upper right")
plt.savefig("img/RNN-7/perdida")