# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import threading
import time


#################################################################################
#practice

def update():
    for i in range(5):
        print('####in update:current thread is ',threading.current_thread())
        print('####in update:thread count is ', threading.active_count())

        time.sleep(1)#real multi threading,sleep can be interrupt
def ask():
    while (1):
        time.sleep(0.5)
        print('how many threading running?',threading.active_count())
        if 3 == threading.active_count():
            print('yes,update ends,and i will quit~!')
            break
        print('newly threading count:', threading.active_count())
        print('in main:current thread is ', threading.current_thread())
def main():
    print('called main()')
    print(threading.active_count())
    #print(threading.enumerate())#what's wrong
    print(threading.current_thread())
    #Thread(func())wrong!didn't create a new threading,just called update() once

    ########################demo start#######################################
    #demo0:default:real multi threading,
    #but not real wait(if there is no while loop) thread ends
    thread1 = threading.Thread(target=update)
    thread1.start()
    while (1):
        time.sleep(0.2)
        print('does update threading ends?')
        if 2 == threading.active_count():
            print('yes,update ends,and i will quit~!')
            break
        print('newly threading count:', threading.active_count())
        print('in main:current thread is ', threading.current_thread())

    #demo1:join finally,no effect
    # because main function didn't execute join yet
    '''
    thread1 = threading.Thread(target=update)
    thread1.start()

    while (1):
        time.sleep(0.2)
        print('does update threading ends?')
        if 2 == threading.active_count():
            print('yes,update ends,and i will quit~!')
            break
        print('newly threading count:', threading.active_count())
        print('in main:current thread is ', threading.current_thread())

    thread1.join()
'''
    #demo2:join firstly,and main function just wait in this step
    '''
    thread1 = threading.Thread(target=update)
    thread1.start()
    thread1.join()
    while (1):
        time.sleep(0.2)
        print('does update threading ends?')
        if 2 == threading.active_count():
            print('yes,update ends,and i will quit~!')
            break
        print('newly threading count:', threading.active_count())
        print('in main:current thread is ', threading.current_thread())
'''
    #demo3:another way to realize real multi threading
    '''
    thread1 = threading.Thread(target=update)
    thread2 = threading.Thread(target=ask)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
'''


    ###################demo end##############
    print('main finish')


if __name__ == '__main__':
    print(threading.active_count())
    print(threading.current_thread())
    main()

