# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:34:28 2017

@author: chenlonghua
"""

import tensorflow as tf 

#create constant op 
a = tf.constant(2)
b = tf.constant(3)

#launch the default graph
with tf.Session() as sess:
    print('Addition with constants:%i' % sess.run(a+b))
    print('Mutiplication with constants:%i' % sess.run(a*b))
    
    
    
    
#tf grahp input 
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.ini16)

#define some operations 
add =tf.add(a,b)
mul =tf.multiply(a,b)

#launch the default graph
with tf.Session() as sess:
    print('Addition with variables:%i' % sess.run(add,feed_dict={a:2,b:3}))
    print('Multiplication with variables:%i' % sess.run(mul,feed_dict={a:2,b:3}))
    
    
#create constant matrix    
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

#multiplication
product = tf.matmul(matrix1,matrix2)


with tf.Session() as sess:
    result = sess.run(product)
    print(result)