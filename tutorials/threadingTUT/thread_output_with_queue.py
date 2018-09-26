# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial


import threading
import time
from queue import Queue

def func(l,q):
    for i in range(len(l)):
        l[i] = l[i] ** 2
    q.put(l)
    print(l)
def main_thread():
    q = Queue()
    input_list = [[1,2,3],[4,5,6],[3,2,1],[2,2,3]]
    threads = []
    results = []
    for i in range(len(input_list)):
        t = threading.Thread(target=func,args=(input_list[i],q))
        t.start()
        #t.join()
        threads.append(t)
        #print(i)
    for t_iter in threads:
        t_iter.join()

    for i in range(len(threads)):
        results.append(q.get())
    print(results)

if __name__ == '__main__':
    main_thread()



###############################
