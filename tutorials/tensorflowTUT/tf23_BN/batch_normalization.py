import tensorflow as tf
import numpy as np



ACTIVATION = tf.nn.relu
#activation_function(Wx_plus_b) equals to tf.nn.relu(Wx_plus_b)
N_LAYERS = 7
N_HIDDEN_UNITS = 30


def built_net(xs,ys,norm):
    def add_layer(inputs,in_size,out_size,activation_function=None,norm=False):
        weights = tf.Variable(tf.random_normal([in_size,out_size],
                                               mean=0.0,stddev=1.))
#two way to add biases:way 1
##        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
##        Wx_plus_b = tf.matmul(inputs,weights)+biases
#way2
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        Wx_plus_b = tf.nn.bias_add(tf.matmul(inputs,weights),biases)
        #########################################################################3
        
        if(norm):
            fc_mean,fc_variance = tf.nn.moments(Wx_plus_b,axes=[0])
            scale = tf.Variable(tf.ones([out_size]))
            shift = tf.Variable(tf.zeros([out_size]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean),tf.identity(fc_var)
            mean, var = mean_var_with_update()

            Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b,mean,var,shift,scale,epsilon)
            
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs
    ##########func end#################
    if(norm):
        # BN for the first input
        fc_mean, fc_var = tf.nn.moments(
            xs,
            axes=[0],
        )
        scale = tf.Variable(tf.ones([1]))
        shift = tf.Variable(tf.zeros([1]))
        epsilon = 0.001
        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = mean_var_with_update()
        xs = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

    #layers_inputs[N_LAYERS]
    #layers_inputs[xs]#why????????mistake!is = [xs] not layers_inputs[xs]
    layers_inputs = [xs]
    #this is a batch data,batch size is xs,and this is only for input
    #other layer is inserted
    
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value
        #columns's num or axis=1's count,meanwhile,output's width
        
        layer_out = add_layer(layer_input,
                              in_size,
                              N_HIDDEN_UNITS,
                              ACTIVATION,
                              norm)
        layers_inputs.append(layer_out)

    #final_out = layers_out
    #hidden_units->logits
    prediction = add_layer(layers_inputs[-1],30,1,activation_function=None)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys),reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op,cost,layers_inputs]

#############################################################3
#make up some data
#fix_seed(1)#tf and np  seed
x_data = np.linspace(-7,10,2500)[:,np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0,8,x_data.shape)
#x  (-7,10)
#y=x^2 - 5
#y  (-5,95)
y_data = np.square(x_data) - 5 + noise

###############################################################3
#plot data
###############################################################33
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
train_op,cost,layers_inputs = built_net(xs,ys,norm=False)
train_op_norm,cost_norm,layers_inputs_norm = built_net(xs,ys,norm=True)
#sess
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(250):

        if i % 50 == 0:    
            all_inputs,all_inputs_norm = sess.run(
                [layers_inputs,layers_inputs_norm],{xs:x_data,ys:y_data})
            #print(all_inputs.shape)
            #print(all_inputs_norm.shape)
        sess.run([train_op,train_op_norm],
##                 {xs:x_data[i*10:(i+1)*10],ys:y_data[i*10:(i+1)*10]})
                {xs:x_data[i*10:i*10+10],ys:y_data[i*10:i*10+10]})

    
#train
#test















