import numpy as np

a = np.array([[1,2,3],
              [4,5,6]])
a2 = np.array([[[1],[2],[3]],
               [[4],[5],[6]]])
b = np.array([10,20,30])
b2 = np.array([[10],[20],[30]])
print(a.shape)
print(a2.shape)
print(b.shape)
print(b2.shape)
c = [100]
d = 100

print(a+b)
#print('a+b2:',a+b2)
print('a2+b:\n',a2+b)
print('a2+b2:\n',a2+b2)
print(a+c)
print(a+d)