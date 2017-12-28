# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:25:22 2017

@author: chenlonghua
"""

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets(r'C:\Users\chenlonghua.JW\Documents\GitHub\tensorflow_learn\data',one_hot=True)

#parameters 
learning_rate =0.01 
training_epochs = 25 
batch_size =100 
display_step =1 


#tf grahp input 
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#set model weights w和b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#construct model 构建模型
pred = tf.nn.softmax(tf.matmul(x,W) + b)

#损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),axis=1) )
#优化函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#初始化参数
init = tf.global_variables_initializer() 

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost =0 
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,
                                                       y:batch_ys})
            #compute average loss 
            avg_cost +=c / total_batch 
            
        if (epoch+1) % display_step ==0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print('Optimization Finished!')
    
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    
    
    
    
    
    
    
    
    
    