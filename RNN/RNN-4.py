import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.compat.v1.Session()

x_vals = np.linspace(0,10,100)
y_vals = x_vals + np.random.normal(0,1,100)

plt.clf()
plt.plot(x_vals, y_vals, 'bo', label = "Datos")
plt.savefig("img/RNN-4_pruido")

x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1,100)))
A = np.column_stack((x_vals_column, ones_column))
b = np.transpose(np.matrix(y_vals))

A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

tA_A = tf.matmul(tf.transpose(A_tensor),A_tensor)
L = tf.linalg.cholesky(tA_A)
tA_b = tf.matmul(tf.transpose(A_tensor),b_tensor)
y_sol = tf.linalg.solve(L, tA_b)
x_sol = tf.linalg.solve(tf.transpose(L),y_sol)

sol_eval = sess.run(x_sol)
print(sol_eval)

slope = sol_eval[0][0]
intercept = sol_eval[1][0]
print(str(np.round(slope, 3))+"x+"+str(np.round(intercept,3)))

best_fit = []
for i in x_vals:
    best_fit.append(slope*i+intercept)

plt.clf()
plt.plot(x_vals, y_vals, 'bo', label = "Datos")
plt.plot(x_vals, best_fit, 'r-', label = "Regresión lineal con LU", linewidth = 3)
plt.legend(loc="upper left")
plt.savefig("img/RNN-4_sol")