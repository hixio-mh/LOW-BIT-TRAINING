import numpy as np
import functools
import tensorflow as tf

from time import time
from avatar import Avatar
import dynamic_fixed_point as dfxp

avatar = Avatar()

def average_gradients(tower_grads):
    avg_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(tf.stack(grads), axis=0)
        v = grad_and_vars[0][1]
        avg_grads.append((grad, v))
    return avg_grads


def tower_reduce_mean(towers):
    return tf.reduce_mean(tf.stack(towers), axis=0)

def write_file(file_name, num, b, h, w, c):
    print(np.array(num).shape)
    file = open(file_name, 'w')
    for i in range(b):
        for j in range(h):
            for m in range(w):
                for n in range(c):
                    file.write(str('%.6f' % num[i][j][m][n]))
                    file.write('\r\n')

def write_file_fc(file_name, num, b, h):
    print(np.array(num).shape)
    file = open(file_name, 'w')
    for i in range(b):
        for j in range(h):
            file.write(str('%.6f' % num[i][j]))
            file.write('\r\n')                  

def write_file_soft(file_name, num, b):
    print(np.array(num).shape)
    file = open(file_name, 'w')
    for i in range(b):
        file.write(str('%.6f' % num[i]))
        file.write('\r\n') 

class LearningRateScheduler:
    def __init__(self, lr, lr_decay_epoch, lr_decay_factor):
        self.lr = tf.get_variable('learning_rate', initializer=lr)
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor
        self.epoch = tf.get_variable('lr_scheduler_step',
            dtype=tf.int32, initializer=0)

        update_epoch = tf.assign(self.epoch, self.epoch+1)
        with tf.control_dependencies([update_epoch]):
            self.update_lr = tf.assign(self.lr, tf.cond(
                tf.equal(tf.mod(self.epoch, self.lr_decay_epoch), 0),
                lambda : self.lr * self.lr_decay_factor,
                lambda : self.lr,
            ))

    def step(self):
        '''
        Op for updating learning rate.

        Should be called at the end of an epoch.
        '''
        return self.update_lr


class Trainer:
    def __init__(self, model, dataset, dataset_name, logger, params):

        self.n_epoch = params.n_epoch
        self.exp_path = params.exp_path

        self.logger = logger

        self.graph = tf.Graph()
        with self.graph.as_default():
            global_step = tf.train.get_or_create_global_step()

            self.lr_scheduler = LearningRateScheduler(params.lr, params.lr_decay_epoch, params.lr_decay_factor)
            optimizer = tf.train.MomentumOptimizer(self.lr_scheduler.lr, params.momentum)

            tower_grads, tower_loss = [], []

            with tf.variable_scope(tf.get_variable_scope()):
                #net = model(params.bits, params.dropout, params.weight_decay, params.stochastic, training=True)

                #tf.get_variable_scope().reuse_variables()
                
                #self.logger.info('\n' + net.info())

                images = avatar.batch_data()
                images = tf.cast(tf.reshape(images, [-1,28,28,1]), dtype=tf.float32)
                lables = avatar.batch_lable()

                #logits, _ = net.forward(images)
                #loss = tf.nn.softmax_cross_entropy_with_logits(labels=lables, logits=logits)
                #loss = tf.reduce_mean(loss)
                #net.backward(tf.gradients(loss, logits)[0])

                #tower_grads.append(net.grads_and_vars())
                #tower_loss.append(loss)

                conv =  dfxp.Conv2d_q(name='conv', bits=params.bits, training = False, ksize=[5, 5, 1, 20], strides=[1, 1, 1, 1], padding='VALID')
                self.conv_in = images
                self.batch_in = conv.forward(images)
                batch = dfxp.Normalization_q(name='batch', bits=params.bits, num_features=20, training=True)
                self.scale_in = batch.forward(self.batch_in)
                scale = dfxp.Rescale_q(name='scale', bits=params.bits, training=False, num_features=20)
                self.relu_in = scale.forward(self.scale_in)
                relu = dfxp.ReLU_q()
                self.pool_in = relu.forward(self.relu_in)
                pool = dfxp.MaxPool_q(ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
                flat = dfxp.Flatten_q(12*12*20)
                self.fc1_in = pool.forward(self.pool_in)
                self.flat = flat.forward(self.fc1_in)
                fc1 = dfxp.Dense_q(name='dense1', bits=params.bits, training = False, in_units=12*12*20, units=100)
                self.fc2_in = fc1.forward(self.flat)
                fc2 = dfxp.Dense_q(name='dense2', bits=params.bits, training = False, in_units=100, units=10)
                self.softmax_in = fc2.forward(self.fc2_in)
                self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=lables, logits=self.softmax_in)
                self.softmax_indiff = tf.gradients(self.loss, self.softmax_in)
                self.fc2_indiff = fc2.backward(self.softmax_indiff[0], False)



            #grads_and_vars = average_gradients(tower_grads)
            #self.train_op = optimizer.apply_gradients(grads_and_vars, global_step)
            #self.train_loss = tower_reduce_mean(tower_loss)
            #self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            self.summary = tf.summary.merge_all()
            self.graph.finalize()

    def train(self):

        self.logger.info('Start of training')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = False
        config.log_device_placement = False

        with tf.Session(config=config, graph=self.graph) as sess:
            
            self.logger.info('Initializing variables..')
            sess.run(self.init_op)

            for epoch in range(self.n_epoch):
                self.logger.info("*******************************" + str(epoch) + "*******************************")

                r1 = sess.run([self.fc2_indiff])
                print(np.array(r1).shape)
                write_file_fc("softmax_indiff_q.txt", r1[0], 16, 10)

                sess.run(self.lr_scheduler.step())


