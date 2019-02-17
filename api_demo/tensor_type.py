import tensorflow as tf
a = tf.placeholder("float",name = 'a_name')
b = tf.placeholder("float", name = 'b')
y = a*b
y2 = tf.multiply(a,b,name = 'multiply')
sess = tf.Session()
print(a)
print(y)
print(type(a))
print(type(y))
#print(tf.get_default_graph().get_operation_by_name('a'))
#print(tf.get_default_graph().get_tensor_by_name('a_name'))
print('operations:',tf.get_default_graph().get_operations)
operations = tf.get_default_graph().get_operations()
print('name:',operations[0].name)
print(operations[0].type)
print('name:',operations[1].name)
print(operations[1].type)
print('name:',operations[2].name)
print(operations[2].type)
print('name:',operations[3].name)
print(operations[3].type)

print(sess.run(y,{a:3,b:4}))
print(sess.run(y2,{a:3,b:4}))


###############################################################3
zxc = 5
print(zxc)
jkl = [zxc]
print(jkl)
print(jkl[0])

sess.close()
