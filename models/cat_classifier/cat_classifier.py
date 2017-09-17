# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:12:24 2017

@author: dmare
"""

import tensorflow as tf
import matplotlib.pyplot as plt


from cat_nets.datasets.generate_csv import generate_CatClassification_csv
from cat_nets.datasets.read_pets import catClassification_loader


NUM_CLASSES = 12



def inference(images):
    pass






graph = tf.Graph()


with graph.as_default():
    train_path = '..\\..\\cat_nets\\datasets\\cat_pet_train.csv'
    image, label, cat_dict = catClassification_loader(train_path)
    
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.expand_dims(image,0)
    image = tf.map_fn(lambda im: tf.image.per_image_standardization(im), image)
    image = tf.squeeze(image)
    

    
    
    
    
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    
    

with tf.Session(graph = graph) as sess:
    
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    im,name = sess.run([image,label])
    
    plt.imshow(im)
    plt.title(cat_dict[name])
    coord.request_stop()
    coord.join(threads)