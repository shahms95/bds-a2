import tensorflow as tf

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        # "c220g5-120102.wisc.cloudlab.us:2222"
        "localhost:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "localhost:2223"
        # "c220g5-120102.wisc.cloudlab.us:2223"
    ],
    "worker" : [
        "c220g5-120102.wisc.cloudlab.us:2222",
        "c220g5-120118.wisc.cloudlab.us:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "localhost:2223"
        # "c220g5-120102.wisc.cloudlab.us:2223"
    ],
    "worker" : [
        "c220g5-120102.wisc.cloudlab.us:2222",
        "c220g5-120118.wisc.cloudlab.us:2222",
        "c220g5-120106.wisc.cloudlab.us:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
server.join()
