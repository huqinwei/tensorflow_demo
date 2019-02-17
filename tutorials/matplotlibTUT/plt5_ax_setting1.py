# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 5 - axis setting
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://www.scipy-lectures.org/intro/matplotlib/matplotlib.html
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2)
# plot the second curve in this figure with certain parameters
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
# set x limits
plt.xlim((-2.5,2.8))
plt.ylim((-8,15))#curve stays,graph became larger
plt.xlabel('input_data')
plt.ylabel('goodness')

# set new sticks
new_ticks = np.linspace(-3,4,8)#set larger than xlim,and the graph became larger
print(new_ticks)
plt.xticks(new_ticks)
# set tick labels
plt.yticks([-5, -1.8, -1, 1.22, 6],
           ['really\ bad', 'bad', r'$normal$', r'$good$', r'$really\ good$'])


plt.show()
