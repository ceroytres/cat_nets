from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import tensorflow as tf
import csv




def catClassification_loader(path):
    cat_names = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
             'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
             'Siamese','Sphynx']
    cat_dict = dict(zip(cat_names,range(len(cat_names))))
    
    labels_list, filename_list = [], []
    
    with open(path,mode = 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            labels_list.append(cat_dict[row[0]])
            filename_list.append(row[1])
    
    labels_list = tf.convert_to_tensor(labels_list)
    images_list = tf.convert_to_tensor(filename_list)
    

    
    filename_queue = tf.train.slice_input_producer([labels_list,images_list], shuffle=True)
    
    label = filename_queue[0]
    filename = filename_queue[1]
    
    raw_image = tf.read_file(filename)
    image = tf.image.decode_jpeg(raw_image, channels = 3)
    
    
    cat_dict = dict(zip(cat_dict.values(),cat_dict.keys()))
    return image, label, cat_dict

#    image = tf.image.resize_images(image,image_size,
#                                   method = tf.image.ResizeMethod.BILINEAR,
#                                   align_corners= True)
#    image = tf.cast(image, tf.uint8)
#       
#    batch_size = batch_size
#
#    capacity = min_after_dequeue + 3 * batch_size
#    
#    image_batch, label_batch = tf.train.shuffle_batch([image,label], 
#                                              batch_size = batch_size,
#                                              capacity = capacity,
#                                              min_after_dequeue = min_after_dequeue,
#                                              num_threads=num_threads)
#    return image_batch,label_batch
