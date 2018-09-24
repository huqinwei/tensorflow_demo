# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 4 - figure
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
y22 = x**2+2
y23 = x**2+3
#y3 = x**3

plt.figure()
plt.plot(x, y2)
plt.plot(x, y22)

plt.figure()
plt.plot(x,y2)

plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y1)
# plot the second curve in this figure with certain parameters
plt.plot(x, y2, color='red', linewidth=5.0, linestyle='--')
plt.plot(x, y22, color='black', lw=2.0, linestyle='--')
plt.plot(x, y23, c='red', lw=3.0, ls='--')
plt.show()
