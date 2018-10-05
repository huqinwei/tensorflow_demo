import tensorflow as tf
import os

if not os.path.exists('read'):
    os.makedirs('read/')



# #demo1
# with tf.Session() as sess:
#     filename = ['A.jpg', 'B.jpg', 'C.jpg']
#     #print(type(filename))#list
#     #print((filename))
#     #filename_queue = tf.train.string_input_producer(string_tensor = filename, shuffle = True)#no effect
#     #filename_queue = tf.train.string_input_producer(string_tensor = filename, shuffle = True, num_epochs= 5)
#     filename_queue = tf.train.string_input_producer(string_tensor = filename, shuffle = False, num_epochs= 4)
#
#     reader = tf.WholeFileReader()
#     key, value = reader.read(filename_queue)  # read a tuple every time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     print(key)
#     print(value)
#
#     tf.local_variables_initializer().run()
#
#     threads = tf.train.start_queue_runners(sess = sess)
#
#     i = 0
#
#     while True:
#         print('i is %d'   %i)
#         i += 1
#         #print(sess.run(key))#just run 8 times,less than 15!!!!!!!!!!!!15 = len(filename) * num_epochs
#         #print(sess.run(value))#just run 8 times,less than 15
#         #conclusion:every run key or value consume a queue data!!!!!!!!!!!!!!
#         #final sess.run(value) will raise an errorrrrrrrrrr,so loop_time = len(filename) * num_epochs + 1
#         #key,value all stands for an op of read
#
#         image_data = sess.run(value)
#         print('read done!')
#         #key,image_data = sess.run([key,value])
#         #key,image_data = sess.run((key,value))
#         #key,image_data = sess.run(value)#too many values to unpack
#         #see demo 2
#         with open('read/test_%d.jpg' % i, 'wb') as f:
#             f.write(image_data)

#demo2
# with tf.Session() as sess:
#     filename = ['A.jpg', 'B.jpg', 'C.jpg']
#     filename_queue = tf.train.string_input_producer(string_tensor=filename, shuffle=False, num_epochs=4)
#     reader = tf.WholeFileReader()
#     k_v_tuple = reader.read(filename_queue)  # read a tuple every time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     tf.local_variables_initializer().run()
#     threads = tf.train.start_queue_runners(sess=sess)
#     i = 0
#     while True:
#         print('i is %d' % i)
#         i += 1
#         key, image_data = sess.run(k_v_tuple)
#         print(key)
#         with open('read/test_%d.jpg' % i, 'wb') as f:
#             f.write(image_data)

#demo3 non tuple in queue?
#why key is necessary???????????????????????
#can read (key)?yes,because this is string_input_producer
#no demo with value input-output!!!!!!!!!!!!!!!!!
#queue is black box!!!!???????
# tf.train.input_producer
# with tf.Session() as sess:
#     filename = ['A.jpg', 'B.jpg', 'C.jpg']
#     filename = [2,2,3,3,3,3]
#     filename_queue = tf.train.input_producer(input_tensor = filename, shuffle = False, num_epochs= 4)
#
#
#     # reader = tf.WholeFileReader()
#     # key, value = reader.read(filename_queue)  # read a tuple every time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#     out = tf.assign(filename_queue,10)
#     tf.local_variables_initializer().run()
#     threads = tf.train.start_queue_runners(sess = sess)
#
#     i = 0
#     while True:
#         print('i is %d'   %i)
#         i += 1
#
#         value = sess.run(out)
#         print(value)
#
#         # with open('read/test_%d.jpg' % i, 'wb') as f:
#         #     f.write(image_data)


#demo4 not start queue,dead loop,
# with tf.Session() as sess:
#     filename = ['A.jpg', 'B.jpg', 'C.jpg']
#     filename_queue = tf.train.string_input_producer(string_tensor = filename, shuffle = False, num_epochs= 4)
#
#     reader = tf.WholeFileReader()
#     key, value = reader.read(filename_queue)  # read a tuple every time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     print(key)
#     print(value)
#
#     tf.local_variables_initializer().run()
#
#     i = 0
#
#     while True:
#         print('i is %d'   %i)
#         i += 1
#
#         image_data = sess.run(value)
#         print('read done!')
#         with open('read/test_%d.jpg' % i, 'wb') as f:
#             f.write(image_data)


#demo5   use FixedLengthRecordReader
#to test if while-loop times will increase in FixedLengthRecordReader condition:yes,it is!!!!!!!!!!
#demo5.2   #does the key is same?NO!
#WholeFile:b'A.jpg'
#Fixed:b'A.jpg:n'  n is iter counts
# with tf.Session() as sess:
#     filename = ['A.jpg', 'B.jpg', 'C.jpg']
#     filename_queue = tf.train.string_input_producer(string_tensor = filename, shuffle = False, num_epochs= 1)
#
#     #reader = tf.FixedLengthRecordReader(20)#demo 5
#     reader = tf.WholeFileReader()#demo 5
#     key_value = reader.read(filename_queue)  # read a tuple every time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
#     tf.local_variables_initializer().run()
#     tf.train.start_queue_runners(sess=sess)
#
#     i = 0
#
#     while True:
#         print('i is %d'   %i)
#         i += 1
#         key_,value_ = sess.run(key_value)
#         print("key is ",key_)
#         print(" and value is ",value_, ", read done!")
#         # with open('read/test_%d.jpg' % i, 'wb') as f:
#         #     f.write(image_data)

#demo6
#Can not convert a FIFOQueue into a Tensor or Operation.
with tf.Session() as sess:
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    filename_queue = tf.train.string_input_producer(string_tensor = filename, shuffle = False, num_epochs= 1)

    #reader = tf.FixedLengthRecordReader(20)#demo 5

    tf.local_variables_initializer().run()
    tf.train.start_queue_runners(sess=sess)

    i = 0

    while True:
        print('i is %d'   %i)
        i += 1
        value_ = sess.run(filename_queue)

        print(" and value is ",value_, ", read done!")
        # with open('read/test_%d.jpg' % i, 'wb') as f:
        #     f.write(image_data)










