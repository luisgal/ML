import os
import tensorflow as tf
import skimage.io as imd
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
import random



"""Carga de imagenes por directorio"""
def load_images(data_directory):
    dirs = [d for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory,d))]

    labels = []
    images = []
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]

        for f in file_names:
            images.append(imd.imread(f))
            labels.append(int(d))
    return images, labels

"""Guardar png con 6 ejemplos de iagenes"""
def save_exm_img(images, labels, name, cmap="brg"):
    rand_signs = random.sample(range(0, len(labels)), 6)
    plt.figure()
    for i in range(len(rand_signs)):
        temp_im = images[rand_signs[i]]
        plt.subplot(3, 2, i + 1)
        plt.title("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()), fontsize=10)
        plt.axis("off")
        plt.imshow(temp_im, cmap=cmap)
        plt.subplots_adjust(wspace=1, hspace=0.5)

    plt.savefig(name)
    plt.clf()

"""Ejemplos por clase"""
def exm_for_class(images, labels, name):
    unique_labels = set(labels)
    plt.figure(figsize=(16, 16))
    i = 1
    for label in unique_labels:
        temp_im = images[list(labels).index(label)]
        plt.subplot(8, 8, i)
        plt.axis("off")
        plt.title("Clase {0} ({1})".format(label, list(labels).count(label)))
        i += 1
        plt.imshow(temp_im)
    plt.savefig(name)
    plt.clf()




main_dir = "../datasets/belgian/"
train_dir = os.path.join(main_dir,"Training")
test_dir = os.path.join(main_dir,"Testing")

images, labels = load_images(train_dir)

images = np.array(images)
labels = np.array(labels)

plt.figure()
plt.hist(labels, len(set(labels)))
plt.savefig("RNN-1-1")
plt.clf()




save_exm_img(images, labels, "RNN-1_img_originales")

exm_for_class(images,labels,"RNN-1_ejemplos_originales")




w_min = 9999
h_min = 9999
w_max = 0
h_max = 0

for image in images:
    if image.shape[0] < h_min:
        h_min = image.shape[0]
    if image.shape[1] < w_min:
        w_min = image.shape[1]
    if image.shape[0] > h_max:
        h_max = image.shape[0]
    if image.shape[1] > w_max:
        w_max = image.shape[1]
print("Tamaño mínimo: {0}x{1}".format(h_min,w_min))
print("Tamaño mínimo: {0}x{1}".format(h_max,w_max))

images30 = [transform.resize(image, (30,30)) for image in images]
save_exm_img(images30, labels, "RNN-1_img_transformadas")

images30 = np.array(images30)
images30 = rgb2gray(images30)
save_exm_img(images30, labels, "RNN-1_img_gray", "gray")

"""
with tf.Session() as sess:
    print(sess.run(res))
"""