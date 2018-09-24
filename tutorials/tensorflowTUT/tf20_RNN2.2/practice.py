import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LR = 0.006

INPUT_SIZE = 1
OUTPUT_SIZE = 1

BATCH_START = 0
BATCH_SIZE = 50
STEP_SIZE = 20

CELL_SIZE = 10#HIDDEN,self.cell_size

def get_batch():
    global BATCH_START,TIME_STEP
    
    x = np.arange(
        BATCH_START,BATCH_START + BATCH_SIZE * STEP_SIZE
        ).reshape(BATCH_SIZE,STEP_SIZE)/(10 * np.pi)
    seq = np.sin(x)
    res = np.cos(x)
    BATCH_START += STEP_SIZE
    
##    plt.plot(x[0, :], res[0, :], 'r', x[0, :], seq[0, :], 'b--')
##    plt.show()
##    plt.pause(0.3)
    
    #why[seq,res,x],if no [],it looks like ok
    #shape,need trans?~~~~~~~~~~~~~~~~~!!!!!!!!!!!!expand,new axis's num=1
    return seq[:,:,np.newaxis],res[:,:,np.newaxis],x

class LSTMRNN(object):
    def __init__(self,batch_size,step_size,input_size,cell_size,output_size):
        #self.inputs = inputs
        self.step_size = step_size
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.output_size = output_size
        self.input_size = input_size

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, step_size, input_size])#sequence
            self.ys = tf.placeholder(tf.float32, [None, step_size, output_size])#result

        #need variable_scope,because of get_variable
        with tf.variable_scope('input_layer'):
            self.add_input_layer()
        with tf.variable_scope('lstm_layer'):#does this need variable_scope?
            self.add_cell()
        with tf.variable_scope('output_layer'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):        
            self.train_step = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        #reshape:batch,step,input->batch*step,input
        layer_in_x = tf.reshape(self.xs, [-1,self.input_size], name='3_2D')

        #Wx_plus_b
        Ws_in = self._weight_variable([self.input_size,self.cell_size])
        bs_in = self._bias_variable([self.cell_size])
        with tf.name_scope('Wx_plus_b'):
            layer_in_y = tf.matmul(layer_in_x,Ws_in)+bs_in
        
        #reshape:batch*step,input->batch,step,n_hidden_units
        self.layer_in_y = tf.reshape(layer_in_y,
                                     [-1, self.step_size, self.cell_size],
                                     name='2_3D')

    def add_cell(self):
        #tf.contrib.rnn.BasicLSTMCell   tf.nn.rnn_cell.BasicRNNCell
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size,
                                               forget_bias = 1.0,
                                               state_is_tuple = True,
                                               name = 'LSTMCell_by_huqw',
                                               #dtype=tf.float32
            )
        with tf.name_scope('initial_state'):
            self.cell_init_state = rnn_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.outputs, self.final_state = tf.nn.dynamic_rnn(rnn_cell,
                                           self.layer_in_y,
                                            initial_state=self.cell_init_state,
                                           time_major=False,
                                           #dtype=tf.float32,
                                           #scope=None
                                                      )

    def add_output_layer(self):
        #shape:batch*steps,cell_size
        #outputs = self.outputs[-1]
        layer_out_x = tf.reshape(self.outputs,[-1,self.cell_size],name='2_2D')
        
        #reshape:batch,step,hidden->batch*step,hidden
        #Wx_plus_b
        weights = self._weight_variable([self.cell_size,self.output_size])
        biases = self._bias_variable([self.output_size])
        #compute prediction
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(layer_out_x,weights)+biases


    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            #b*x's predictions
            logits = [tf.reshape(self.pred,[-1],name='reshape_pred')],#2d,shape(batch_size,num_decoder_symbols AKA one_hot n_classes)
            targets = [tf.reshape(self.ys,[-1],name='reshape_targets')],#1d,batch_size
            weights = [tf.ones([self.batch_size * self.step_size],dtype=tf.float32)],#1d,batch_size
            average_across_timesteps = True,#dont' understand
            softmax_loss_function = self.ms_error,
            name = 'sequance_loss_by_exmple_by_huqw'
                  )
        with tf.name_scope('average_cost'):
            #self.cost = tf.reduce_sum(losses) / self.batch_size
            self.cost = tf.div(tf.reduce_sum(losses) , self.batch_size, name='average_cost')
            tf.summary.scalar('cost',self.cost)

    @staticmethod#for comute cost
##    def ms_error(pred,ys):        #error:params name does matter!!!!!
##        return tf.square(tf.subtract(pred, ys))
    def ms_error(labels,logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self,shape,name='weights'):
        initializer = tf.random_normal_initializer(mean=0.,stddev=1.,)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)

    def _bias_variable(self,shape,name='biases'):
        initializer = tf.random_normal_initializer(0.1)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)






if __name__ == '__main__':

    model = LSTMRNN(#inputs,
                    batch_size = BATCH_SIZE,
                    step_size = STEP_SIZE,
                    input_size = INPUT_SIZE,
                    cell_size = CELL_SIZE,
                    output_size = OUTPUT_SIZE
                    )
    #train
    with tf.Session() as sess:
        plt.ion()
        plt.show()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs',sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(200):
            seq, res, xs = get_batch()
            
            
            if i == 0:#initial state
                feed_dict = {
                    model.xs:seq,
                    model.ys:res,
                    #state:initial_state
                    }
            else:#last final state
                feed_dict = {
                    model.xs:seq,
                    model.ys:res,
                    model.cell_init_state:final_state}
                
            _,cost,final_state,pred = sess.run(
                [model.train_step,model.cost,model.final_state,model.pred],#todo,from model
                                      feed_dict=feed_dict)


##            plt.plot(xs[0,:], res[0].flatten(), 'r',
##            xs[0,:], pred.flatten()[:STEP_SIZE], 'b--')

            plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:STEP_SIZE], 'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.3)



            
            if i % 20 == 0:
                
                print('cost:',round(cost,4))
                result = sess.run(merged, feed_dict)#merge what???
                writer.add_summary(result, i)


