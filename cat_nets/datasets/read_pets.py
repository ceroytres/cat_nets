from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange


graph = tf.Graph()


path_to_cats = './cat_classification.csv'

with graph.as_default():
    filenames = tf.train.match_filenames_once(path_to_cats)
    filename_queue = tf.train.string_input_producer(filenames)
    
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    
    record_defaults = [[""],[""]]
    label, filename = tf.decode_csv(value,record_defaults)
    
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image)
    
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session(graph=graph) as sess:
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    
    im = sess.run(image)
    plt.imshow(im)
    plt.show()
    
    coord.request_stop()
    coord.join(threads)