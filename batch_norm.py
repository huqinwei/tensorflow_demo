import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
def create_var(name, shape, initializer, trainable = True):
    return tf.get_variable(name, shape = shape, dtype = tf.float32,
                           initializer = initializer, trainable = trainable)
# batch norm layer
def batch_norm(x, decay=0.999, epsilon=1e-03, is_training=True,
               scope="scope"):
    x_shape = x.get_shape()
    
    num_inputs = x_shape[-1]#最低维度的尺寸——通道数
    reduce_dims = list(range(len(x_shape) - 1))#按channel做平均。（只是为了求平均值，输出当然还是原维度）
    print('reduce_dims:',reduce_dims)
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        beta = create_var("beta", [num_inputs,],
                               initializer=tf.zeros_initializer())
        gamma = create_var("gamma", [num_inputs,],
                                initializer=tf.ones_initializer())
        # for inference
        moving_mean = create_var("moving_mean", [num_inputs,],
                                 initializer=tf.zeros_initializer(),
                                 trainable=False)
        moving_variance = create_var("moving_variance", [num_inputs],
                                     initializer=tf.ones_initializer(),
                                     trainable=False)
    if is_training:
        mean, variance = tf.nn.moments(x, axes=reduce_dims)
        print('mean and variance:', mean, variance)
        update_move_mean = moving_averages.assign_moving_average(moving_mean,
                                                mean, decay=decay)
        update_move_variance = moving_averages.assign_moving_average(moving_variance,
                                                variance, decay=decay)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)#添加到更新操作
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
    else:
        mean, variance = moving_mean, moving_variance
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon),beta,gamma,mean,variance,moving_mean,moving_variance
