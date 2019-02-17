import numpy as np
import matplotlib.pyplot as plt
import time
BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
#    plt.ioff()
#    plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
#    plt.ion()
#    plt.show()
#    print("sleep")
#    time.sleep(0.5)
#    plt.ion()
    #returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]



##plt.ion()
##for i in range(15):
##    print('start')
##    seq,res,xs = get_batch()
##    plt.plot(xs[0,:], res[0,:],xs[0,:],seq[0,:],'b--')
##    print('pause')
##    plt.pause(1)
##    plt.show()
##    print('pause')
##    plt.close()
    
fig = plt.figure()
ax = fig.add_subplots(3,3)
plt.ion()
plt.ioff()

for i in range(15):
    print('start')
    seq,res,xs = get_batch()
    plt.plot(xs[0,:], res[0,:],xs[0,:],seq[0,:],'b--')
    print('pause')
    plt.pause(1)
    plt.show()
    print('pause')
    plt.close()
        
