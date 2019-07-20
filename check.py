import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data

import images as img

#Load The dataset
data=input_data.read_data_sets("temp/data/",one_hot="true")


with tf.Session()  as sess:
        saver = tf.train.import_meta_graph('G:\PycharmProjects\class\mnst\DDL.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('G:\PycharmProjects\class\mnst/'))

        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        Y = graph.get_tensor_by_name("Y:0")
        layer_out = graph.get_tensor_by_name("Optimizer:0")
        im=sess.run(layer_out,feed_dict={X:img.x})
        print(np.argmax(im))

