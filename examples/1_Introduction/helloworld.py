# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:30:36 2017

@author: chenlonghua
"""

import tensorflow as tf 

#create a constant op 
hello = tf.constant('hello,tensorflow!')

#start tf session 
sess = tf.Session()
#run the op  
print(sess.run(hello))