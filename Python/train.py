#-*- coding: UTF-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from avatar import Avatar
import numpy as np

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
avatar = Avatar()

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
lable = tf.placeholder(tf.float32, shape=[None, 10], name='lalbe')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean = 0, stddev = 0.1)
  return tf.Variable(initial)

def weight_variable1(shape):
  data = open("weight.txt", 'r')
  imgs = []
  lines = data.readlines()
  num = 0
  for line in lines:
    for db in line.split():
      imgs.append(float(db))
      num += 1
      if num == shape:
        break
  x = np.array(imgs).astype(np.float32)
  tx = tf.convert_to_tensor(x)
  initial = tf.reshape(tx, [5, 5, 1, 20])
  return tf.Variable(initial)

def weight_variable2(shape):
  data = open("weight.txt", 'r')
  imgs = []
  lines = data.readlines()
  num = 0
  for line in lines:
    for db in line.split():
      imgs.append(float(db))
      num += 1
      if num == shape:
        break
  x = np.array(imgs).astype(np.float32)
  tx = tf.convert_to_tensor(x)
  initial = tf.reshape(tx, [12*12*20, 100])
  return tf.Variable(initial)

def weight_variable3(shape):
  data = open("weight.txt", 'r')
  imgs = []
  lines = data.readlines()
  num = 0
  for line in lines:
    for db in line.split():
      imgs.append(float(db))
      num += 1
      if num == shape:
        break
  x = np.array(imgs).astype(np.float32)
  tx = tf.convert_to_tensor(x)
  initial = tf.reshape(tx, [100, 10])
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')

W_conv1 = weight_variable1(5 * 5 * 1 * 20)
b_conv1 = bias_variable([20])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = conv2d(quantize(x_image), quantize(W_conv1))
h_norm1 = tf.layers.batch_normalization(quantize(h_conv1), center=True, scale=True, epsilon=1e-5, training = True)
h_relu1 = tf.nn.relu(h_norm1)
h_pool1 = max_pool_2x2(h_relu1)

W_fc1 = weight_variable2(12 * 12 * 20 * 100)
b_fc1 = bias_variable([100])
h_pool2_flat = tf.reshape(h_pool1, [-1, 12*12*20])
h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

W_fc2 = weight_variable3(100 * 10)
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels=lable, logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy1)

train_step1 = tf.train.MomentumOptimizer(0.01, 0.9)
train_step = train_step1.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(lable,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#save权值变量
saver = tf.train.Saver()  

sess.run(tf.global_variables_initializer())
for i in range(1):
  #batch = mnist.train.next_batch(50)
  batch_data = avatar.batch_data()
  batch_lable = avatar.batch_lable()

  train_step.run(feed_dict={x: batch_data, lable: batch_lable})
  result = sess.run(cross_entropy, feed_dict={x: batch_data, lable: batch_lable})
  print(np.array(result).shape)
  print(result)

  #if i%100 == 0:
  #  train_accuracy = accuracy.eval(feed_dict={x: batch[0], lable: batch[1]})
  #  print("step %d, training accuracy:%g"%(i, train_accuracy))

  #train_step.run(feed_dict={x: batch[0], lable: batch[1]})