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
#train_path,_,_ = generate_CatClassification_csv([.7,.2,.1])








graph = tf.Graph()


with graph.as_default():
    image, label, cat_dict = catClassification_loader(train_path)
    
    image = tf.image.convert_image_dtype(image,tf.float32)
    

    
    
    isTraining = tf.constant(True)
    
    
    x_in = tf.placeholder(tf.float32,[None,None,None,3])
    
    x = tf.cond(isTraining, lambda : image, lambda : x_in )
    

    y = tf.cond(isTraining, lambda: tf.one_hot(label, depth = NUM_CLASSES),
                lambda: _ )
    
    x_feed = tf.map_fn(lambda im: tf.image.per_image_standardization(im), x)
    
    
    with tf.variable_scope("conv1"):
        
        W = tf.get_variable("W",shape = [5,5,64,128],
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable("b",shape = [128],
                            initializer= tf.truncated_normal_initializer())
        
        h1 = tf.nn.elu(tf.nn.conv2d(x_feed,W,[1,1,1,1],'SAME'))
    
    
    
    
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    
    

with tf.Session(graph = graph) as sess:
    
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    im,name = sess.run([x_feed,label])
    
    plt.imshow(im)
    plt.title(cat_dict[name])
    coord.request_stop()
    coord.join(threads)