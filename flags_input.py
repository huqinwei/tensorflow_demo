#tf.app.flags  demo (equals to argv)
import tensorflow as tf

tf.app.flags.DEFINE_string('name', 'foo', 'this is a arg')
tf.app.flags.DEFINE_integer('age', 18, 'this is your age')
tf.app.flags.DEFINE_boolean('marry', False, 'did you marrys')

FLAGS = tf.app.flags.FLAGS

#def main():#TypeError:takes 0 but 1 was given
def main(_):

    print(FLAGS.name)
    print(FLAGS.age)
    #print(FLAGS.asssge)#AttributeError
    print(FLAGS.marry)
if __name__ == '__main__':
    tf.app.run()













