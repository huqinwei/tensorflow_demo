# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import threading
import time

#demo1:default
'''
def job1():
    global A, lock
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 1
        print('job1', A)
    lock.release()

def job2():
    global A, lock#

    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 10
        print('job2', A)
    lock.release()

if __name__ == '__main__':
    lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    '''
#demo2:success too,no need to declare global lock
'''
def job1():
    global A#, #lock
    print(type(lock),'and id is:',id(lock))
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 1
        print('job1', A)
    lock.release()

def job2():
    global A#, lock  #no need to declare global lock?
    #indeed,global declare is not necessary!!!!!!!!
    #https://www.cnblogs.com/summer-cool/p/3884595.html
    #why can lock,but can not A?????????????????
    #they are diffent type,hard to say!!!!!!!!
    #object is special!same name,but not same id!!!!!!!!!!

    print(type(lock),'and id is:',id(lock))
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 10
        print('job2', A)
    lock.release()

if __name__ == '__main__':
    lock = threading.Lock()
    print(type(lock),'and id is:',id(lock))
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(type(lock),'and id is:',id(lock))
    #lock.acquire()
    print('in main,A is :',A)
'''

#demo2.4:lock obj can be overwrite in local,but is not really overwrite
#local lock object is different from main's lock object
#however,this will leads to lock didn't working!!!!
'''
def job1():
    global A#, #lock
    lock = threading.Lock()#is not error!but this leads to lock not working!!
    print(type(lock),'and id is:',id(lock))
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 1
        print('job1', A)
    lock.release()

def job2():
    global A#, lock  #no need to declare global lock?
    #indeed,global declare is not necessary!!!!!!!!
    #https://www.cnblogs.com/summer-cool/p/3884595.html
    #why can lock,but can not A?????????????????
    #they are diffent type,hard to say!!!!!!!!
    #object is special!same name,but not same id!!!!!!!!!!

    lock = threading.Lock()#is not error!but this leads to lock not working!!
    print(type(lock),'and id is:',id(lock))
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 10
        print('job2', A)
    lock.release()

if __name__ == '__main__':
    lock = threading.Lock()
    print(type(lock),'and id is:',id(lock))
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(type(lock),'and id is:',id(lock))
    #lock.acquire()
    print('in main,A is :',A)

'''







#demo3:failure!local lock
'''
def job1():
    global A#, #lock
    lock = threading.Lock()
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 1
        print('job1', A)
    lock.release()

def job2():
    global A#, lock 
    
    lock = threading.Lock()
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 10
        print('job2', A)
    lock.release()

if __name__ == '__main__':
    #lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
'''

#demo4:deperately declare global lock
#not allowed!!!!!!!!!!!!
'''
def job1():
    global A#, #lock
    global lock = threading.Lock()#not allowed!!!!!!!!!!!!
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 1
        print('job1', A)
    lock.release()

def job2():
    global A#, lock  #no need to declare global lock?
    global lock = threading.Lock()#not allowed!!!!!!!!!!!!
    lock.acquire()
    for i in range(10):
        time.sleep(0.1)
        A += 10
        print('job2', A)
    lock.release()

if __name__ == '__main__':
    A = 0
    t1 = threading.Thread(target=job1)
    t2 = threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
'''