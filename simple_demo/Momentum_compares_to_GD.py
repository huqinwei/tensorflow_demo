import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
COMMON_LR = 0.01


#simplified fixed gradient to Z = X * X + 50 * Y * Y ,which is Z' = 2X + 100Y
def g(x):
    return np.array([2 * x[0], 100 * x[1]])
# x = [2,3]
# y = g(x)
# print(len(x))
# print(y.shape)
# print(y)

#make up data
x = np.linspace(-200, 200, 1000)
y = np.linspace(-100, 100, 1000)
X,Y = np.meshgrid(x, y)#可以这么理解，meshgrid函数用两个坐标轴上的点在平面上画网格。
Z = X * X + 50 * Y * Y

print(x.shape)
print(y.shape)
print('X.shape:', X.shape)
print('X.flatten.shape:', X.flatten().shape)
print('Y.shape:', Y.shape)
print('Z.shape:', Z.shape)

# %matplotlib inline
def contour(X,Y,Z, arr = None):
    plt.figure(figsize=(15,7))
    xx = X.flatten()
    yy = Y.flatten()
    zz = Z.flatten()
    plt.contour(X, Y, Z, colors='black')
    # plt.contour(X, Y, colors='black')#ValueError: Contour levels must be increasing
    plt.plot(0,0,marker='*')
    if arr is not None:
        arr = np.array(arr)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i+2,0],arr[i:i+2,1])
# contour(X,Y,Z)
# plt.show()

def gd(x_start, step, g):#gradient descent
    x = np.array(x_start, dtype='float64')
    # print(x)
    passing_dot = [x.copy()]#training record
    for i in range(50):
        grad = g(x)
        x -= grad * step

        passing_dot.append(x.copy())
        print('[ epoch {0} ]   grad={1},   x={2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:#early stop
            break
    return x, passing_dot

# res, x_arr = gd([150,75], 0.009, g)
res, x_arr = gd([150,75], 0.016, g)
# res, x_arr = gd([150,75], 0.019, g)
# res, x_arr = gd([150,75], 0.020, g)#cannot convergence
# res, x_arr = gd([150,75], 0.023, g)
# contour(X,Y,Z, x_arr)


def momentum(x_start, step, g, discount = 0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)

    for i in range(50):
        grad = g(x)
        pre_grad = pre_grad * discount + grad
        x -= pre_grad * step
        passing_dot.append(x.copy())

        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i,grad,x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot

res,x_arr = momentum([150,75],0.016,g)
# contour(X,Y,Z,x_arr)

# 1.Nesterov是Momentum的变种。
# 2.与Momentum唯一区别就是，计算梯度的不同，Nesterov先用当前的速度v更新一遍参数，在用更新的临时参数计算梯度。
# 3.相当于添加了矫正因子的Momentum。
# 4.在GD下，Nesterov将误差收敛从O（1/k），改进到O(1/k^2)
# 5.然而在SGD下，Nesterov并没有任何改进

def nesterov(x_start, step, g, discount = 0.7):
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    pre_grad = np.zeros_like(x)
    for i in range(50):
        x_future = x - step * discount * pre_grad#!!!!!!!!!!!!!!!!!!!temp update
        grad = g(x_future)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!grad in temp update point
        #??????????????????????????????????????????????????????
        pre_grad = pre_grad * 0.7 + grad#!!!!!!!!!!!!!!!!!!!0.7 should be discount?????????????????????????????????????
        #????????????????????????????????????????????????
        x -= pre_grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i,grad,x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot
res, x_arr = nesterov([150,75], 0.012, g)
contour(X,Y,Z,x_arr)

















plt.show()










############################################################################

#demo#no different with contourf()
# x = np.linspace(-200, 200, 10)
# y = np.linspace(-100, 100, 10)
# X,Y = np.meshgrid(x, y)#可以这么理解，meshgrid函数用两个坐标轴上的点在平面上画网格。
# Z = X * X + 50 * Y * Y
# # h = plt.contourf(x,y,Z)
# h = plt.contourf(X,Y,Z)
# plt.show()

#######################################
#demo different with plot()
# x = np.linspace(-200, 200, 10)
# y = np.linspace(-100, 100, 10)
# X,Y = np.meshgrid(x, y)#可以这么理解，meshgrid函数用两个坐标轴上的点在平面上画网格。
# Z = X * X + 50 * Y * Y
#
# plt.plot(X,Y, marker='.', color='red', linestyle='none')#is a matrix
# # plt.plot(x,y, marker='.', color='blue', linestyle='none')#is a line
# plt.show()

#######################################
#demo???????????????
# x = np.linspace(-200, 200, 10)
# y = np.linspace(-100, 100, 10)
# X,Y = np.meshgrid(x, y)#可以这么理解，meshgrid函数用两个坐标轴上的点在平面上画网格。
# Z = X * X + 50 * Y * Y
#
# # plt.plot(X,Y,Z, marker='.', color='blue', linestyle='none')
# plt.plot(x,y,Z, marker='.', color='blue', linestyle='none')
# plt.show()

#######################################
# h = plt.contourf(x,y,Z)
# plt.show()
# h = plt.contourf(X,Y,Z)
# plt.show()

# plt.axis([-100, 100, -100, 100])
# plt.plot(x, z)
# plt.show()
# print(z)


#Momentum
# tf.train.MomentumOptimizer(COMMON_LR).minimize()

#Gradient Descent

# tf.train.GradientDescentOptimizer(COMMON_LR).minimize()


with tf.Session() as sess:
    tf.global_variables_initializer().run()




