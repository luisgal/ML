import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.compat.v1.Session()

#Carga y manipulación de datos
data_iris = datasets.load_iris()
binary_target = np.array([1.0 if x==0 else 0.0 for x in data_iris.target])
iris_2d = np.array([[x[2], x[3]] for x in data_iris.data])

#Variables
batch_size = 20
x1_data = tf.placeholder(shape=[None,1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None,1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None,1], dtype=tf.float32)

#Modelo   x_2 = A*x_1 + b  -->  x_2 - A*x_1 - b = 0
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

product = tf.matmul(x2_data,A)
sum = tf.add(product,b)
prediction = tf.subtract(x1_data, sum)

#Funcion de perdidias y propagacion para atras
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y_target)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_step = optimizer.minimize(xentropy)


#Inicializar variables
init = tf.global_variables_initializer()
sess.run(init)

#Entrenar el modelo
for i in range(1000):
    rand_idx = np.random.choice(len(iris_2d),size=batch_size)
    rand_x = iris_2d[rand_idx]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_idx]])
    sess.run(train_step, feed_dict={x1_data:rand_x1,
                                    x2_data:rand_x2,
                                    y_target:rand_y})
    if (i+1)%100==0:
        print("Paso #{0}\tA: {1}\tb: {2}".format(str(i+1),str(sess.run(A)),str(sess.run(b))))



[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

x = np.linspace(0, 3, num=100)
abline_values = []
for i in x:
    abline_values.append(slope * i + intercept)

setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 1]

no_setosa_x = [a[1] for i, a in enumerate(iris_2d) if binary_target[i] == 0]
no_setosa_y = [a[0] for i, a in enumerate(iris_2d) if binary_target[i] == 0]

plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='Setosa')
plt.plot(no_setosa_x, no_setosa_y, 'ro', label = "No setosa")
plt.plot(x, abline_values, 'b-')
plt.suptitle('Separación lineal de las Setosas')
plt.xlabel("Longitud de los Pétalos")
plt.ylabel("Anchura de los Pétalos")
plt.xlim([0,3])
plt.ylim([0,10])
plt.savefig("img/iris")


