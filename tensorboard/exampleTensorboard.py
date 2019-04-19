import tensorflow as tf
import os

g = tf.Graph()

with g.as_default():
    m1 = tf.constant([[3., 3.]], name="mat-1")
    m2 = tf.constant([[2.], [4.]], name="mat-2")
    p = tf.matmul(m1, m2, name="product")


    sess = tf.Session()
    result = sess.run(p)
    tf.summary.FileWriter("%s/exampleTensorboard" % (os.environ.get("TF_LOG_DIR")), sess.graph)
    sess.close()
    print result
