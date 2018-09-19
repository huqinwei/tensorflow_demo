import tensorflow as tf
a = 3
c = 1
if a==3:
#	c = 5
	print(c)

print(c)

name = 'global0'
def f1():
	print(name)
def f2():
	name = 'frank2'
	return f1
ret = f2()
ret()

li = [lambda :x for x in range(10)]
print(type(li))
print((li))
print(type(li[0]))
print((li[0]()))
print((li[1]()))
print((li[2]()))
print((li[3]()))




name = 'global'
def f1():
	age = 18
	global name
	name = 'local'
	print(age,name)
f1()
print(name)

