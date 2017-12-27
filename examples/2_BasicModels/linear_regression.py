# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:46:18 2017

@author: chenlonghua
"""

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#参数
learning_rate =0.01 
training_epochs =1000 
display_step =50 

#训练数据
train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

#占位符
X = tf.placeholder(tf.float32 )
Y = tf.placeholder(tf.float32 )

#权重 w和b
W = tf.Variable(np.random.randn(),name='weight')
b = tf.Variable(np.random.randn(),name='bias')

#模型
pred = tf.add(tf.multiply(X,W), b )

cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)


#损失函数，l2
cost = tf.reduce_mean(tf.reduce_sum(tf.square(pred - Y)))

#优化函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化
init = tf.global_variables_initializer() 

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
            
        if (epoch +1 ) %display_step ==0:
            c = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))
    print('Optimization Finished!')
    training_cost = sess.run(cost,feed_dict ={X:train_X,y:train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

            
            