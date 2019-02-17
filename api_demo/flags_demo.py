import tensorflow as tf
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'tom,jerry',
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')


tf.app.flags.DEFINE_integer(
    'integer_to_print', '99',#legal:99,'99'#illegal:'99.9',99.9
    '!!!!')
tf.app.flags.DEFINE_boolean('boolean_to_print','True','haha')

with tf.variable_scope(name_or_scope = 'tom'):
    w1 = tf.Variable(0, name = 'v1')
    w2 = tf.Variable(1, name = 'v2')
with tf.name_scope(name = 'jerry'):
    w1 = tf.Variable(0, name = 'v1')
    w2 = tf.Variable(1, name = 'v8')
    w3 = tf.Variable(1, name = 'v12', trainable = False)
    w4 = tf.Variable(1, name = 'w12', trainable = True)

variables_all = tf.get_collection(tf.GraphKeys)
variables_trainable_all = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
variables_trainable_tom = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'tom')
variables_trainable_jerry = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope = 'jerry')
print('variables_all:',variables_all)
print('variables_trainable_all:',variables_trainable_all)
print('variables_trainable_tom:',variables_trainable_tom)
print('variables_trainable_jerry:',variables_trainable_jerry)
print('tf.GraphKeys:',tf.GraphKeys)
print('tf.GraphKeys.TRAINABLE_VARIABLES:',tf.GraphKeys.TRAINABLE_VARIABLES)

print(tf.get_variable_scope())
print(tf.get_variable_scope().get_collection(name='v1'))
#print(tf.get_variable_scope(name = 'jerry').get_collection(name='v1'))#TypeError: get_variable_scope() got an unexpected keyword argument 'name'
print(tf.get_collection(key = 'v1', scope = 'tom'))
#print(tf.get_collection( scope = 'tom'))

print(FLAGS.integer_to_print)
print(FLAGS.boolean_to_print)

# Warn the user if a checkpoint exists in the train_dir. Then we'll be
# ignoring the checkpoint anyway.


exclusions = []
if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
print(exclusions)
# TODO(sguada) variables.filter_variables()
variables_to_restore = []
for var in slim.get_model_variables(scope = 'tom'):
    print('hello')
    print(var)
    for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
            break
    else:
        variables_to_restore.append(var)



