import tensorflow as tf
a = tf.placeholder("float",name = 'a_name')
b = tf.placeholder("float")
y = a*b
sess = tf.Session()
print(a)
print(y)
print(type(a))
print(type(y))
#print(tf.get_default_graph().get_operation_by_name('a'))
#print(tf.get_default_graph().get_tensor_by_name('a_name'))
print(tf.get_default_graph().get_operations)
operations = tf.get_default_graph().get_operations()
print(operations[0].name)
print(operations[0].type)
print(operations[1].name)
print(operations[1].type)
print(operations[2].name)
print(operations[2].type)


zxc = 5
print(zxc)
jkl = [zxc]
print(jkl[0])

sess.close()
