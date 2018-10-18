import numpy as np
import tensorflow as tf

#test
# sum = 0
# for i in range(1000):
#     sum += np.random.random()
# print(sum / 1000)

#data produce class

class ToySequenceData(object):
    def __init__(self,n_samples,min_data_len,max_data_len,max_data_value):
        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):
            random_len = np.random.randint(min_data_len, max_data_len)
            # random_len = 8
            # print(random_len)
            self.seqlen.append(random_len)

            if np.random.random() >= 0.5:
                # print('order')
                # second param may be a negative value if max_data_value not big enough
                random_start = np.random.randint(0, max_data_value - random_len)
                # print('start:',random_start)
                data = [(random_start + i) / max_data_value for i in range(random_len)]
                # print(data)
                self.data.append(data)
                self.labels.append([1., 0.])
            else:
                # print('non-order')
                data = [np.random.randint(0,max_data_value) / max_data_value for i in range(3)]
                self.data.append(data)
                self.labels.append([0., 1.])
                # print(data)

        print(len(self.data))
        print(len(self.labels))
        print(len(self.seqlen))
        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))]
        batch_labels = self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.labels))]
        batch_seqlen = self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.seqlen))]
        self.batch_id = min(self.batch_id + batch_size, len(self.data))

        return batch_data, batch_labels, batch_seqlen

t = ToySequenceData(50,3,10,20)
for i in range(10):
    data,labels,seqlen = t.next(8)
    print('data len :\n', len(data))
    print('data:\n', data)
    print(labels)
    print(seqlen)













