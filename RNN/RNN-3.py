import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.compat.v1.Session()

x_vals = np.linspace(0,10,100)
y_vals = x_vals + np.random.normal(0,1,100)

plt.plot(x_vals, y_vals, 'o', label = "Datos")
plt.savefig("img/x_y_ruido")

x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1,100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))

A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

tA_A = tf.matmul(tf.transpose(A_tensor),A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv,tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)
solution_eval = sess.run(solution)

print(solution_eval)

slope = solution_eval[0][0]
intercept = solution_eval[1][0]
print(str(np.round(slope,3))+"x+"+str(np.round(intercept,3)))

best_fit = []
for i in x_vals:
    best_fit.append(slope*i+intercept)

plt.clf()
plt.plot(x_vals, y_vals, 'o', label = "Datos originales")
plt.plot(x_vals, best_fit, 'r-', label = "Recta de regresión lineal", linewidth = 3)
plt.legend(loc = "upper left")
plt.savefig("img/x_y_pred")
