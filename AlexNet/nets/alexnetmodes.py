from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer



def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            opt = configure_optimizer(global_step, total_num_steps)

            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    # for device in devices:
    #     print(device, " ********** ")

    with tf.device(devices[0]):
        builder = ModelBuilder()
        print('num_classes: ' + str(num_classes))
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)

        if not is_train:
            return alexnet_eval(net, labels)

        global_step = builder.ensure_global_step()
        print('total_num_examples: ' + str(total_num_examples))
    train_op = train(total_loss, global_step, total_num_examples)
    return net, logits, total_loss, train_op, global_step


def distribute(images, labels, num_classes, total_num_examples, devices, is_train=True):
    # Put your code here
    # You can refer to the "original" function above, it is for the single-node version.
    # 1. Create global steps on the parameter server node. You can use the same method that the single-machine program uses.
    # 2. Configure your optimizer using HybridMomentumOptimizer.
    # 3. Construct graph replica by splitting the original tensors into sub tensors. (hint: take a look at tf.split )
    # 4. For each worker node, create replica by calling alexnet_inference and computing gradients.
    #    Reuse the variable for the next replica. For more information on how to reuse variables in TensorFlow,
    #    read how TensorFlow Variables work, and considering using tf.variable_scope.
    # 5. On the parameter server node, apply gradients.
    # 6. return required values.
    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""
        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })

    def train(total_loss, global_step, total_num_steps):
        """Build train operations"""
        # Compute gradients
        with tf.control_dependencies([total_loss]):
            print("Value of num_replicas inside train function : {}".format(num_replicas))
            opt = configure_optimizer(global_step, total_num_steps)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([apply_gradient_op]):
            return tf.no_op(name='train')

    # for device in devices:
    #     print(device, " ********** ")

    # with tf.device(tf.train.replica_device_setter(worker_device = "/job:worker/task:%d" % FLAGS.task_index,cluster = clusterinfo)):        
    #     builder = ModelBuilder()
    #     print('num_classes: ' + str(num_classes))
    #     # with tf.variable_scope("scope-{}".format(i)):
    #     net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)

    #     if not is_train:
    #         return alexnet_eval(net, labels)

    images_small = tf.split(images, len(devices)-1)
    labels_small = tf.split(labels, len(devices)-1)
    # if issync:
    #     num_replicas = len(devices)-1
    # else:
    #     num_replicas = 0
    with tf.device(devices[0]):
        builder = ModelBuilder()
        global_step = builder.ensure_global_step()
        print('num_classes: ' + str(num_classes))
        opt = configure_optimizer(global_step, total_num_examples)
        # if num_replicas!=0:
        #     opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_replicas,total_num_replicas=num_replicas)

    # print("Value inside distribute function : {}".format(issync))

    gradients_all = []
    i=0
    for device in devices[:-1]:
        task_index = device[-1]
        # with tf.device(tf.train.replica_device_setter(worker_device=device)):
        with tf.device(device):
            print("------------------------- Device = {} -------------".format(device))
            # builder = ModelBuilder()
            # print('num_classes: ' + str(num_classes))
            with tf.variable_scope("scope-{}".format(i)):
                net, logits, total_loss = alexnet_inference(builder, images_small[i], labels_small[i], num_classes)
                with tf.control_dependencies([total_loss]):
                    # print("Value of num_replicas inside train function : {}".format(num_replicas))
                    grads = opt.compute_gradients(total_loss)
                    gradients_all.append(grads)

            if not is_train:
                return alexnet_eval(net, labels)
        i=i+1
    print('total_num_examples: ' + str(total_num_examples))
    
    #TODO : add code about sv; check Fayi's part2.py for help
    with tf.device(devices[-1]):
        # train_op = train(total_loss, global_step, total_num_examples, num_replicas)
        # Apply gradients.

        print("------------------------- Device = {} -------------".format(devices[-1]))
        # with tf.control_dependencies([grads]):
        #     apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        #     init_token_op = opt.get_init_tokens_op()
        #     chief_queue_runner = opt.get_chief_queue_runner()

        for grad in gradients_all:
            apply_gradient_op = opt.apply_gradients(grad, global_step=global_step)

        # init_token_op = opt.get_init_tokens_op()
        # chief_queue_runner = opt.get_chief_queue_runner()
        # init = tf.global_variables_initializer()
        # task_index = devices[-1][-1]
        # sv = tf.train.Supervisor(is_chief=(task_index==0),init_op=init,summary_op=None, global_step=global_step)
        # sess = sv.prepare_or_wait_for_session(server.target)
        # if task_index == 0:
            # sv.start_queue_runners(sess, [chief_queue_runner])
            # sess.run(init_token_op)

        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

    return net, logits, total_loss, train_op, global_step

