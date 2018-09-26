# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import threading
from queue import Queue
import copy
import time


def compute(l,q):
    print('compute id:',id(l))
    q.put(sum(l))
    #print('sleep')
    #time.sleep(1)

def multithreading(l):
    print('multi id:',id(l))
    q = Queue()
    threads = []
    res = 0
    for i in range(4):
        print('i = ',i)
        #important:need copy l!!!!!!!
        t = threading.Thread(target=compute,args=(copy.copy(l),q))
        t.start()
        threads.append(t)
        #t.join()#wrong:this will wait for a thread to complete!

    [t.join for t in threads]
    for th in threads:
        res += q.get()
    print(res)

def normal(l):
    print('normal id:',id(l))
    res = sum(l)
    print(res)

if __name__ == '__main__':
    l = list(range(1000000))
    print('main l id:',id(l))
    s_t = time.time()
    normal(l*4)
    print('after normal main l id:',id(l))
    print('normal: ',time.time()-s_t)
    s_t = time.time()
    multithreading(l)
    print('multithreading: ', time.time()-s_t)
'''
    l = list(range(10))
    print(l)
    print(l*4)
    print(sum(l))
    print(sum(l*4))
'''
