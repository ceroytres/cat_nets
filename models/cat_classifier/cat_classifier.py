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
BATCH_SIZE = 128


graph = tf.Graph()

with graph.as_default():
    train_path = '..\\..\\cat_nets\\datasets\\cat_pet_train.csv'
    image, label, cat_dict = catClassification_loader(train_path)

    
    image = tf.image.convert_image_dtype(image,tf.float32)
    image = tf.expand_dims(image,0)
    image = tf.image.resize_bicubic(image,[32,32],
                                             align_corners= True)
    
    mu,var = tf.nn.moments(image,[1,2])
    image = (image-mu)/tf.sqrt(var)
    image = tf.squeeze(image)
    
    image = tf.reshape(image,(32*32*3,))
    target  = tf.one_hot(label,NUM_CLASSES)


    
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3* BATCH_SIZE
    image_batch, target_batch = tf.train.shuffle_batch([image,target],
                                                       batch_size= BATCH_SIZE,
                                                       capacity=capacity,
                                                       min_after_dequeue
                                                       =min_after_dequeue)
    
    
    W = tf.get_variable("W", shape=(32*32,), initializer=tf.glorot_normal_initializer())
    b = tf.get_variable("b", shape=(32,), initializer=tf.constant_initializer(0))
    
    
    logits = tf.nn.xw_plus_b(image_batch,W,b)
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=target_batch)

    opt = tf.train.AdamOptimizer().minimize(ce_loss)    
    
    
    
    
    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    
    

with tf.Session(graph = graph) as sess:
    
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    im,t = sess.run([image_batch,target_batch])
    
    coord.request_stop()
    coord.join(threads)