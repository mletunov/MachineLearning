import numpy as np
import tensorflow as tf
import dataset.utils as utils
from learning.baseVideo import HockeyNetwork

class CnnVideoNetwork(HockeyNetwork):
    def build(self, rnn_state=20, num_steps=30, learning_rate=1e-2, l2_loss=False, avg_result=False):
        num_classes = 2
        frame_size = (288, 360, 3)

        tf.reset_default_graph()
        tf.set_random_seed(100)

        with tf.name_scope("input"):
            # num_steps sequence of frames: height x width x depth
            x = tf.placeholder(tf.float32, (None, num_steps, *frame_size))
            # video class - fight (1) or not (0)
            y = tf.placeholder(tf.int32, (None,))
        
            flatten_x = tf.reshape(x, (-1, *frame_size))

        with tf.name_scope("cnn"):
            with tf.name_scope("conv1"):
                conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 16), stddev=0.1), name="w")
                conv1_b = tf.Variable(tf.zeros(shape=(16)), name="b")
                conv1 = tf.nn.relu(tf.nn.conv2d(flatten_x, conv1_w, strides=[1, 2, 2, 1], padding="VALID") + conv1_b, name="conv")
                
            with tf.name_scope("conv2"):
                conv2_w = tf.Variable(tf.truncated_normal(shape=(7, 7, 16, 16), stddev=0.1), name="w")
                conv2_b = tf.Variable(tf.zeros(shape=(16)), name="b")
                conv2 = tf.nn.relu(tf.nn.conv2d(conv1, conv2_w, strides=[1, 2, 2, 1], padding="VALID") + conv2_b, name="conv")
               
            with tf.name_scope("conv3"):
                conv3_w = tf.Variable(tf.truncated_normal(shape=(9, 9, 16, 16), stddev=0.1), name="w")
                conv3_b = tf.Variable(tf.zeros(shape=(16)), name="b")
                conv3 = tf.nn.relu(tf.nn.conv2d(conv2, conv3_w, strides=[1, 2, 2, 1], padding="VALID") + conv3_b, name="conv")
                pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="VALID", name="pool")
            
            with tf.name_scope("flatten"):
                size = np.prod(pool3.get_shape().as_list()[1:])
                flatten_conv = tf.reshape(pool3, shape=(-1, size))

        with tf.name_scope("rnn"):            
            rnn_inputs = tf.reshape(flatten_conv, (-1, num_steps, size))
            cell = tf.nn.rnn_cell.GRUCell(rnn_state)
            init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            output = tf.reshape(rnn_outputs, shape=(-1, rnn_state)) if avg_result else rnn_outputs[:,-1,:]

            with tf.name_scope("dense"):
                dense_w = tf.Variable(tf.truncated_normal([rnn_state, num_classes]), tf.float32, name="w")
                dense_b = tf.Variable(tf.zeros([num_classes]), tf.float32, name = "b")
                dense = tf.add(tf.matmul(output, dense_w), dense_b, name="out")

        with tf.name_scope("prediction"):
            score = tf.reduce_mean(tf.reshape(dense, shape=(-1, num_steps, num_classes)), axis = 1) if avg_result else dense
            prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)
            prediction_one_hot = tf.one_hot(prediction, depth=num_classes)
            prediction_total = tf.reduce_mean(prediction_one_hot, axis=0)

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

        with tf.name_scope("train"):
            if l2_loss:
                loss = tf.nn.l2_loss(score - tf.one_hot(y, depth=num_classes))
            else:
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(score, y)
            
            total_loss = tf.reduce_mean(loss)
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

        return dict(
                x=x,
                y=y,
                prediction=prediction,
                prediction_total=prediction_total,
                accuracy=accuracy,
                loss=total_loss,
                train_step=train_step,
                frames_count=num_steps)

    def train(self, graph, dataset, epochs, batch_size=20):
        np.random.seed(100)
        datasets = super()._datasets(
                            lambda: utils.split(dataset.read_names(), frac = 0.9), 
                            lambda train_names: dataset.gen_dataset(train_names, by_video=False, frames_count=graph['frames_count'], batch_size=batch_size),
                            lambda test_names: dataset.gen_dataset(test_names, by_video=True, frames_count=graph['frames_count'], batch_size=batch_size))

        return super()._train(graph=graph, epochs=epochs, **datasets)