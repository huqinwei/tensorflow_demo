import tensorflow as tf

#run with parameter --string1 'goodbye',then print string 'goodbye'

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('string1','hello','help?')
#if FLAGS.checkpoint_path is None:

if FLAGS.checkpoint_path is None:
    print('is None')
print(FLAGS.checkpoint_path)
print(FLAGS.string1)
#print(FLAGS.string1.attr)
print(type(FLAGS.string1))

print(type(FLAGS))
