import tensorflow as tf

#keep_prob = 0.5
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
input_data = tf.placeholder(tf.float32,shape=[None,224,224,3], name='input')
#input_data = tf.placeholder(tf.float32,shape=[1,224,224,3], name='input')
y_ = tf.placeholder(tf.float32,[2],"realLabel")#what for in this example?

#1-1	224*224*3->224*224*64		[3,3,1,64]	b	[64]
with tf.variable_scope("conv1_1"):
    kernel1_1 = tf.Variable(tf.truncated_normal([3,3,3,64]), dtype=tf.float32)
    bias1_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64], name='BIAS1_1'))#0.1 is better?
    conv1_1 = tf.nn.conv2d(input_data, kernel1_1, strides=[1,1,1,1], padding='SAME', name="CONV1_1")
    conv1_1_plus_biases = tf.nn.bias_add(conv1_1, bias1_1)
    conv1_1_relu = tf.nn.relu(conv1_1_plus_biases)

#1-2	same				[3,3,64,64]
with tf.variable_scope("conv1_2"):
    kernel1_2 = tf.Variable(tf.truncated_normal([3,3,64,64]), dtype=tf.float32)
    bias1_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64], name='BIAS1_2'))#0.1 is better?
    conv1_2 = tf.nn.conv2d(conv1_1_relu, kernel1_2, strides=[1,1,1,1], padding='SAME', name="CONV1_2")
    conv1_2_plus_biases = tf.nn.bias_add(conv1_2, bias1_2)
    conv1_2_relu = tf.nn.relu(conv1_2_plus_biases)

#pooling	112*112*64		[1,2,2,1]	b	none
max_pool1 = tf.nn.max_pool(conv1_2_relu, [1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool1')
#2-1	112*112*128			[3,3,64,128]		[64]
with tf.variable_scope("conv2_1"):
	kernel2_1 = tf.Variable(tf.truncated_normal([3,3,64,128]), dtype=tf.float32)
	bias2_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128], name='BIAS2_1'))#0.1 is better?
	conv2_1 = tf.nn.conv2d(max_pool1, kernel2_1, strides=[1,1,1,1], padding='SAME', name="CONV2_1")
	conv2_1_plus_biases = tf.nn.bias_add(conv2_1, bias2_1)
	conv2_1_relu = tf.nn.relu(conv2_1_plus_biases)

#2-2					[3,3,128,128]	
with tf.variable_scope("conv2_2"):
	kernel2_2 = tf.Variable(tf.truncated_normal([3,3,128,128]), dtype=tf.float32)
	bias2_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128], name='BIAS2_2'))#0.1 is better?
	conv2_2 = tf.nn.conv2d(conv2_1_relu, kernel2_2, strides=[1,1,1,1], padding='SAME', name="CONV2_2")
	conv2_2_plus_biases = tf.nn.bias_add(conv2_2, bias2_2)
	conv2_2_relu = tf.nn.relu(conv2_2_plus_biases)

#pooling	56*56*128
max_pool2 = tf.nn.max_pool(conv2_2_relu, [1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool2')

#3-1		56*56*256		[3,3,128,256]
with tf.variable_scope("conv3_1"):
    kernel3_1 = tf.Variable(tf.truncated_normal([3,3,128,256]), dtype=tf.float32)
    bias3_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256], name='BIAS3_1'))#0.1 is better?
    conv3_1 = tf.nn.conv2d(max_pool2, kernel3_1, strides=[1,1,1,1], padding='SAME', name="CONV3_1")
    conv3_1	= tf.nn.bias_add(conv3_1, bias3_1)
    conv3_1_relu = tf.nn.relu(conv3_1)
					
#3-2					[3,3,256,256]
with tf.variable_scope("conv3_2"):
    kernel3_2 = tf.Variable(tf.truncated_normal([3,3,256,256]), dtype=tf.float32)
    bias3_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256], name='BIAS3_2'))#0.1 is better?
    conv3_2 = tf.nn.conv2d(conv3_1_relu, kernel3_2, strides=[1,1,1,1], padding='SAME', name="CONV3_2")
    conv3_2	= tf.nn.bias_add(conv3_2, bias3_2)
    conv3_2_relu = tf.nn.relu(conv3_2)

#3-3
with tf.variable_scope("conv3_3"):
    kernel3_3 = tf.Variable(tf.truncated_normal([3,3,256,256]), dtype=tf.float32)
    bias3_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256], name='BIAS3_3'))#0.1 is better?
    conv3_3 = tf.nn.conv2d(conv3_2_relu, kernel3_3, strides=[1,1,1,1], padding='SAME', name="CONV3_3")
    conv3_3	= tf.nn.bias_add(conv3_3, bias3_3)
    conv3_3_relu = tf.nn.relu(conv3_3)

#pooling	28*28*256
max_pool3 = tf.nn.max_pool(conv3_3_relu, [1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool3')

#4-1		28*28*512		[3,3,256,512]
with tf.variable_scope("conv4_1"):
    kernel4_1 = tf.Variable(tf.truncated_normal([3,3,256,512]), dtype=tf.float32)
    bias4_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name='BIAS4_1'))#0.1 is better?
    conv4_1 = tf.nn.conv2d(max_pool3, kernel4_1, strides=[1,1,1,1], padding='SAME', name="CONV4_1")
    conv4_1	= tf.nn.bias_add(conv4_1, bias4_1)
    conv4_1_relu = tf.nn.relu(conv4_1)

#4-2
with tf.variable_scope("conv4_2"):
    kernel4_2 = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32)
    bias4_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name='BIAS4_2'))#0.1 is better?
    conv4_2 = tf.nn.conv2d(conv4_1_relu, kernel4_2, strides=[1,1,1,1], padding='SAME', name="CONV4_2")
    conv4_2	= tf.nn.bias_add(conv4_2, bias4_2)
    conv4_2_relu = tf.nn.relu(conv4_2)

#4-3
with tf.variable_scope("conv4_3"):
    kernel4_3 = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32)
    bias4_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name='BIAS4_3'))#0.1 is better?
    conv4_3 = tf.nn.conv2d(conv4_2_relu, kernel4_3, strides=[1,1,1,1], padding='SAME', name="CONV4_3")
    conv4_3	= tf.nn.bias_add(conv4_3, bias4_3)
    conv4_3_relu = tf.nn.relu(conv4_3)

#pooling	14*14*512
max_pool4 = tf.nn.max_pool(conv4_3_relu, [1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool4')

#5-1		14*14*512		[3,3,512,512]	[512]
with tf.variable_scope("conv5_1"):
    kernel5_1 = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32)
    bias5_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name='BIAS5_1'))#0.1 is better?
    conv5_1 = tf.nn.conv2d(max_pool4, kernel5_1, strides=[1,1,1,1], padding='SAME', name="CONV5_1")
    conv5_1	= tf.nn.bias_add(conv5_1, bias5_1)
    conv5_1_relu = tf.nn.relu(conv5_1)

#5-2
with tf.variable_scope("conv5_2"):
    kernel5_2 = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32)
    bias5_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name='BIAS5_2'))#0.1 is better?
    conv5_2 = tf.nn.conv2d(conv5_1_relu, kernel5_2, strides=[1,1,1,1], padding='SAME', name="CONV5_2")
    conv5_2	= tf.nn.bias_add(conv5_2, bias5_2)
    conv5_2_relu = tf.nn.relu(conv5_2)

#5-3
with tf.variable_scope("conv5_3"):
    kernel5_3 = tf.Variable(tf.truncated_normal([3,3,512,512]), dtype=tf.float32)
    bias5_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name='BIAS5_3'))#0.1 is better?
    conv5_3 = tf.nn.conv2d(conv5_2_relu, kernel5_3, strides=[1,1,1,1], padding='SAME', name="CONV5_3")
    conv5_3	= tf.nn.bias_add(conv5_3, bias5_3)
    conv5_3_relu = tf.nn.relu(conv5_3)

#pooling	7*7*512
max_pool5 = tf.nn.max_pool(conv5_3_relu, [1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool5')

#flatten	reshape		[-1,7,7,512]	[-1,25088]
length = max_pool5.get_shape()[3].value * max_pool5.get_shape()[1].value * max_pool5.get_shape()[2].value
#max_pool5_shape = max_pool5.get_shape()
#length = (max_pool5_shape[0].value) * (max_pool5_shape[1].value) * (max_pool5_shape[2].value)
max_pool5_flat = tf.reshape(max_pool5, [-1,length], name='reshape')
#fc6		25088->4096		matmul	[25088,4096]	[4096]
W6 = tf.Variable(tf.truncated_normal([25088,4096]), dtype=tf.float32)
b6 = tf.Variable(tf.constant(0.0,dtype=tf.float32, shape=[4096], name='BIAS6'))
fc6 = tf.matmul(max_pool5_flat, W6)
fc6_plus_biases = tf.nn.bias_add(fc6,b6)
fc6_relu = tf.nn.relu(fc6)
#dropout	keep_prob
fc6_dropout = tf.nn.dropout(fc6_relu,keep_prob, name='fc6_drop')
#fc7		4096
W7 = tf.Variable(tf.truncated_normal([4096,4096]), dtype=tf.float32)
b7 = tf.Variable(tf.constant(0.0,dtype=tf.float32, shape=[4096], name='BIAS7'))
fc7 = tf.matmul(fc6_dropout, W7)
fc7 = tf.nn.bias_add(fc7,b7)
fc7 = tf.nn.relu(fc7)
#dropout
fc7_dropout = tf.nn.dropout(fc7,keep_prob, name='fc7_drop')
#fc8		1000
W8 = tf.Variable(tf.truncated_normal([4096,1000]), dtype=tf.float32)
b8 = tf.Variable(tf.constant(0.0,dtype=tf.float32, shape=[1000], name='BIAS8'))
fc8 = tf.matmul(fc7_dropout, W8)
fc8 = tf.nn.bias_add(fc8,b8)
#important:relu
fc8 = tf.nn.relu(fc8)
#softmax
soft_max = tf.nn.softmax(fc8)
#argmax
predictions = tf.argmax(soft_max,1)

#make up a data
pic = tf.Variable(tf.truncated_normal([1,224,224,3],dtype=tf.float32))
#initial
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter("vgg16_graph/",sess.graph)
#run(softmax)	or	run(prediction)
	print('shape of you:',pic.shape)
	print('type of you:',type(pic))
#	print(sess.run(soft_max,feed_dict={input_data:pic.eval(), keep_prob:1.0}))
#print(sess.run(soft_max,feed_dict={input_data:pic,keep_prob:1.0}))
	print(sess.run(predictions,feed_dict={input_data:pic.eval(), keep_prob:0.5}))
	print(sess.run(predictions,feed_dict={input_data:pic.eval(), keep_prob:1.0}))



