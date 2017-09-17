# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 09:34:03 2017

@author: dmare
"""
from __future__ import print_function
from __future__ import division



import tensorflow as tf




def extract_HOGFeatures(images, orientation_bins = 9, cell_shape = [8,8],
                        sqrt_norm = True):
    
    with tf.name_scope(name, default_name='extract_HOGFeatures', image):
        
        if sqrt_norm:
             images = tf.sqrt(images)
        
        padded_images = tf.pad(images,[[2,2],[2,2],[0,0]], mode= 'SYMMETRIC')
        
        Gx = padded_images[2:-2,2:,:]-padded_images[2:-2,0:,:]
        Gy = padded_images[2:,2:-2,:]-padded_images[0:,2:-2,:]
        
        G = tf.concat([Gx,Gy], axis  = -1)
        
        batch_size = tf.shape(image)
        num_channels = batch_size[-1]
        
        ksizes = [1]+cell_shape+[1]
        stride = [1]+cell_shape+[1]
        
        
        cells = tf.extract_image_patches(G,ksizes = ksizes, stride = stride,
                                         rates = [1,1,1,1])
        cells_shape = tf.shape(cells)
        cells = tf.reshape(cells,[-1,cells_shape[0],cells_shape[1],
                                  cell_shape[0],cell_shape[1],2*num_channels])
    
        cells_x = cells[:,:,:,:,0:(num_channels/2)]
        cells_y = cells[:,:,:,:,(num_channels/2)::]
        
        
        cells_mag = np.sqrt(cells_x**2+cell_y**2)
        cells_angle = tf.atan2(cell_y,cell_x)
        
        
        return None # TODO: Finish
        
        
        