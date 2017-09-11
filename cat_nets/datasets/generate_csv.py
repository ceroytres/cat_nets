# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 01:22:09 2017

@author: dmare
"""
import glob
import os

from six.moves import xrange
from ..utilities import sample_discrete_distribution



def generate_CatClassification_csv(dataset_split):
    """
    Creates csv files for loading the train/validation/test
    
    dataset_split : list of floats 
                    Incidate the probability of a image being placed in 
                    the train,validation, or, test
    
    
    returns:
        train_path : str
                     path to training csv file
        val_path   : str
                     path to val csv file
        test_path  : str
                     path to test csv file
                     
    """
    cat_names = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
                 'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
                 'Siamese','Sphynx']
    
    base_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(base_path,"data\\pet_images\\")
    
    pattern = lambda cat_name : os.path.join(path,cat_name+"*.jpg") 
    filenames = [glob.glob(pattern(cat_name))  for cat_name in cat_names]
    
    train_path = base_path+"\\cat_pet_train.csv"
    test_path = base_path+"\\cat_pet_test.csv"
    val_path = base_path+"\\cat_pet_val.csv"
    
    
    train_writer = open(train_path,"w")
    val_writer = open(val_path,"w")
    test_writer = open(test_path,"w")
    
    for i in xrange(len(cat_names)):
        for j in xrange(len(filenames[i])):
            line = "%s,%s\n" % (cat_names[i], filenames[i][j])
            choice = sample_discrete_distribution(dataset_split)
            
            if choice  == 0:
                train_writer.write(line)
            if choice  == 1:
                val_writer.write(line)
            if choice  == 2:
                test_writer.write(line)
                
    train_writer.close()
    val_writer.close()
    test_writer.close()
    
    return train_path, val_path, test_path

    
    
                
                