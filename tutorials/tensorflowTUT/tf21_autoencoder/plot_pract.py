import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,2*np.pi,400)
y = np.sin(x)
###########################################################
##y2 = np.sin(x**2)
##
##fig,ax=plt.subplots()
##ax.plot(x,y)
##
##fig2,ax2=plt.subplots()
##ax2.plot(x,y2)
######################################################3
##f, (ax1,ax2) = plt.subplots(1,2,sharey=True)
##ax1.plot(x,y)
##ax1.set_title('Sharing Y axis')
###ax1.scatter(x,y)
###ax2.scatter(x,y)
###ax2.scatter(x,y,marker='+')
###ax2.scatter(x,y,marker='o')
##ax2.scatter(x,y,marker='x')
##
##print(type(ax2))
##########################################################
#fig, axes = plt.subplots(3,3,subplot_kw=dict(polar=True))
fig, axes = plt.subplots(3,4)
axes[0,0].plot(x,y)
axes[0,1].scatter(x,y,marker='x')

rand_ = np.random.normal(size=81)
axes[0,2].imshow(np.reshape(rand_,(9,9)),cmap='gray_r')
axes[1,0].imshow(np.reshape(rand_,(9,9)))
axes[1,1].imshow(np.reshape(rand_,(9,9)),cmap='gray')
axes[1,2].imshow(np.reshape(rand_,(9,9)),cmap=plt.cm.binary)

axes[1,3].imshow(np.reshape(np.random.normal(size=900),(30,30)))

axes[2,0].imshow(np.reshape(np.arange(1800,0,-2),(30,30)))
axes[2,1].imshow(np.reshape(np.arange(784),(28,28)))
axes[2,2].imshow(np.reshape(np.arange(784,0,-1),(28,28)))
axes[2,3].imshow(np.reshape(np.arange(784,0,-1),(28,28)),cmap='gray')

#########################################################3
##plt.subplots(2,2,sharex='col')

##plt.subplots(2,2,sharey='row')

##plt.subplots(2,2,sharex='all',sharey='all')

#plt.subplots(3,3,sharex=True,sharey=True)









###########################################################
plt.show()
