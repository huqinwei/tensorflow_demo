# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

#################################################################################
#practice
import threading
import time
def update():
    for i in range(5):
        print('####in update:current thread is ',threading.current_thread())
        print('####in update:thread count is ', threading.active_count())

        time.sleep(4)#real multi threading,sleep can be interrupt


def main():
    print('called main()')
    print(threading.active_count())
    #print(threading.enumerate())#what's wrong
    print(threading.current_thread())
    #thread_added = threading.Thread(target=update())
    # #wrong!didn't create a new threading,just called update() once
    thread_added = threading.Thread(target=update)
    thread_added.start()

    while(1):
        time.sleep(0.2)
        print('does update threading ends?')
        if 2 == threading.active_count():
            print('yes,update ends,and i will quit~!')
            break
        print('newly threading count:', threading.active_count())
        print('in main:current thread is ', threading.current_thread())
if __name__ == '__main__':
    print(threading.active_count())
    print(threading.current_thread())
    main()
