# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 16 - grid
"""
Please note, this script is for python3+.
If you are using python2+, please modify it accordingly.
Tutorial reference:
http://matplotlib.org/users/gridspec.html
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# method 1: subplot2grid
##########################
# plt.figure()
#ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)  # stands for axes
#ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # stands for axes
# ax1 = plt.subplot2grid((3, 3), (0, 1), colspan=3)  # stands for axes
# ax1.plot([1, 1.2, 1.6, 2], [1, 1.1, 1.8, 2])
# ax1.set_title('ax1_title')
# ax1.set_xlabel('x axis label')
# ax1.set_xlim(-1,2)
# ax1.set_ylim(1,2)

#the test after is based on condition that without plt.tight_layout(),otherwize,it will be error
#ax2 = plt.subplot2grid((2, 2), (0, 0), colspan=2)#overwrite ax1
#ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)#not overwrite
#ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)#not overwrite
#ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)#overwrite
#ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=2)#not overwrite!!!!!!!!!!!!!overwrite is just about the exact pixel,not columns and rows

# ax2.set_title(('ax2'))
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax4.scatter([1, 2], [2, 2])
# ax4.set_xlabel('ax4_x')
# ax4.set_ylabel('ax4_y')
# ax5 = plt.subplot2grid((3, 3), (2, 1))
# ax5.set_title(('ax5'))

# method 2: gridspec
#########################
# plt.figure()
# gs = gridspec.GridSpec(3, 3)
# # use index from 0
# #ax6 = plt.subplot(gs[0, :])
# #ax6 = plt.subplot(gs[0, 1])
# #ax6 = plt.subplot(gs[0, :2])
# #ax6 = plt.subplot(gs[0:, 1])
# ax6 = plt.subplot(gs[0:, 0:1])
# ax7 = plt.subplot(gs[1, :2])
# ax8 = plt.subplot(gs[1:, 2])
# ax9 = plt.subplot(gs[-1, 0])
# ax10 = plt.subplot(gs[-1, -2])

# method 3: easy to define structure
####################################
f, ((ax11, jarvis), (ax13, wahaha)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax11.scatter([1,2], [1,2])

ax13.scatter([1,2,3,4,5],[1,2,4,6,7])
wahaha.scatter([1,2,3,4,5],[1,6,4,6,7],c=[1,11,33,22,11])
jarvis.scatter([1,2,3,4,5,6,7,8],[1,6,3,1,4,6,7,5],c=[14,113,33,22,11,1,88,44],cmap='OrRd_r',lw=10)
#jarvis.xlim('jarvis')
#jarvis.xlim(0,10)
jarvis.set_xlim(-1,9)
jarvis.set_title(('jarvis,give me bubble'))
print(type(jarvis))

plt.tight_layout()
plt.show()
