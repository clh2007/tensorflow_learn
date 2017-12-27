# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 14:52:04 2017

@author: chenlonghua
"""

import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(r'C:\Users\chenlonghua.JW\Documents\GitHub\tensorflow_learn\data', one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf Graph Input
#xtr为原数据集
#xte为目标
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
#损失函数：l1,即所有的元素相减
#tf.negative(x) 对x所有元素取负
#tf.add(x,tf.negative(y)) 相当于x-y 
#代价函数：cost
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

l1_accuracy = 0

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xte)):
        nn_index = sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        print("Test",i,"Prediction:",np.argmax(Ytr[nn_index]),
             "True Class:",np.argmax(Yte[i]))
        
        if np.argmax(Ytr[nn_index]) ==np.argmax(Yte[i]):
            l1_accuracy += 1/len(Xte)
            
    print('Done!')
    print('l1 ---Accuracy:',l1_accuracy)
    
    
dis = tf.reduce_sum(tf.sqrt(tf.square(xtr - xte)),reduction_indices=1)
preds = tf.arg_min(dis,0)
l2_accuracy =0
init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init)
    for i in range(len(Xte)):
        nn_index_s = sess.run(preds,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        print("Test",i,"Prediction:",np.argmax(Ytr[nn_index_s]),
             "True Class:",np.argmax(Yte[i]))
        if np.argmax(Ytr[nn_index_s]) ==np.argmax(Yte[i]):
            l2_accuracy += 1/len(Xte)
            
    print('Done!')
    print('l2 --- Accuracy:',l2_accuracy)            