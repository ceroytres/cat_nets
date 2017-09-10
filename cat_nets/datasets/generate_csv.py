# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 01:22:09 2017

@author: dmare
"""
import glob
import os

from six.moves import xrange

cat_names = ['Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
             'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
             'Siamese','Sphynx']

path = ".\data\pet_images"

pattern = lambda cat_name : os.path.join(path,cat_name+"*.jpg") 
filenames = [glob.glob(pattern(cat_name))  for cat_name in cat_names]

file_dict= dict(zip(cat_names,filenames))

with open("cat_classification.csv","w") as f:
    
    for i in xrange(len(cat_names)):
        for j in xrange(len(filenames[i])):
            f.write("%s,%s\n" % (cat_names[i], filenames[i][j]))
    