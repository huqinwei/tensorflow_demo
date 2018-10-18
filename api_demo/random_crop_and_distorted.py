import tensorflow as tf

import numpy as np
tf.set_random_seed(1)
input_for_crop = tf.constant(np.arange(12).reshape(1,3,4))

#image: 4 - D Tensor of shape `[batch, height, width, channels]` or
# 3 - D Tensor of shape `[height, width, channels]`.
input_for_distort = tf.constant(np.arange(12).reshape(3,4,1))

output = tf.random_crop(input_for_crop, size=[1, 2, 3])
output2 = tf.random_crop(input_for_crop, size=[1, 2, 3])
output3 = tf.random_crop(input_for_crop, size=[1, 2, 3])
output4 = tf.random_crop(input_for_crop, size=[1, 2, 3])
output5 = tf.random_crop(input_for_crop, size=[1, 2, 3])

distorted = tf.image.random_flip_left_right(image=input_for_crop)#default axis=0,because this is image's API
distorted2 = tf.image.random_flip_up_down(image=input_for_crop)#
distorted100 = tf.image.random_flip_left_right(image=input_for_distort)#axis=0, length = 1,no effect
distorted101 = tf.image.random_flip_up_down(image=input_for_distort)
distorted102 = tf.image.random_brightness(image=input_for_distort, max_delta=3)
distorted103 = tf.image.random_contrast(image=input_for_distort, lower = 2.2, upper = 7.7)



with tf.Session() as sess:
    print('input:\n',sess.run(input_for_crop))
    print('output:\n',sess.run(output))
    print('output2:\n',sess.run(output2))
    # print(sess.run(output3))
    # print(sess.run(output4))
    # print(sess.run(output5))


    print('distorted:\n',sess.run(distorted))
    print('distorted2:\n',sess.run(distorted2))


    print('distorted100:\n',sess.run(distorted100))
    print('distorted101:\n',sess.run(distorted101))
    print('distorted102:\n',sess.run(distorted102))
    print('distorted103:\n',sess.run(distorted103))

# distorted_image = tf.random_crop(reshaped_image, [height, width, 3])