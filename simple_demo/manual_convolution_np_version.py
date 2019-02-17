import tensorflow as tf
import numpy as np

batch_size = 1

input_height = 5
input_width = 5

filter_height = 3
filter_width = 3

output_height =input_height - filter_height + 1
output_width = input_width - filter_width + 1

input_channel = 3
output_channel = 2

# [batch, in_height, in_width, in_channels]
np_input_arg = np.ones([batch_size, input_height, input_width, input_channel])
# [filter_height, filter_width, in_channels, out_channels]
np_filter_arg = np.ones([filter_height, filter_width, input_channel, output_channel])
np_biases = np.ones([batch_size,1,1,output_channel])

np_final_output = np.zeros([batch_size, output_height, output_width, output_channel])

# manual convolution
for batch in range(batch_size):
    for output in range(output_channel):
        for input in range(input_channel):
            for i in range(output_height):
                for j in range(output_width):
                    # a filter window
                    filter_sum = 0
                    # convolution operation: [i,i+1,i+2] * [j,j+1,j+2]   [3] * [3] = [9]
                    for m in range(filter_height):
                        for n in range(filter_width):
                            np_final_output[batch][i][j][output] += np_input_arg[batch][i + m][j + n][input] * \
                                                                    np_filter_arg[m][n][input][output]
                    if input == input_channel - 1:
                        np_final_output[batch][i][j][output] += np_biases[batch][0][0][output]
                    # print('np_final_output[{0}][{1}][{2}][{3}]:{4}'.format(batch,i,j,output, np_final_output[batch][i][j][output]))
print('np_final_output:', np_final_output)


