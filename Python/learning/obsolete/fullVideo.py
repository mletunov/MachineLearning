import tensorflow as tf
import numpy as np
import dataset.utils as utils
from learning.baseVideo import HockeyNetwork
  
class FullVideoNetwork(HockeyNetwork):       
    def build(self, rnn_state=100, learning_rate=1e-2):
        num_classes = 2
        batch_size = 1

        tf.reset_default_graph()
        tf.set_random_seed(100)

        with tf.name_scope("input"):
            # batch x epoch x height x width x channels; epoch - video frames unknow video length
            x = tf.placeholder(tf.float32, (batch_size, None, 288, 360, 3))
            # video class - fight (1) or not (0)
            y = tf.placeholder(tf.int32, (batch_size, ))

            flatten_x = tf.reshape(x, (batch_size, -1, np.prod(x.get_shape().as_list()[2:])))

        with tf.name_scope("rnn"):
            rnn_inputs = flatten_x
            cell = tf.nn.rnn_cell.GRUCell(rnn_state)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
            output = rnn_outputs[:,-1,:]
    
        with tf.name_scope("dense"):
            dense_w = tf.Variable(tf.truncated_normal([rnn_state, num_classes]), tf.float32, name="w")
            dense_b = tf.Variable(tf.zeros([num_classes]), tf.float32, name = "b")
            dense = tf.add(tf.matmul(output, dense_w), dense_b, name="dense")
        
        with tf.name_scope("prediction"):
            score = dense
            prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)
            prediction_one_hot = tf.one_hot(prediction, depth=num_classes)
            prediction_total = tf.reduce_mean(prediction_one_hot, axis=0)

        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

        with tf.name_scope("train"):
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
                train_step=train_step)

    def train(self, graph, dataset, epochs):
        np.random.seed(100)
        datasets = super()._datasets(
                            lambda: utils.split(dataset.read_names(), frac = 0.9), 
                            lambda train_names: dataset.gen_dataset(train_names, by_video=True),
                            lambda test_names: dataset.gen_dataset(test_names, by_video=True))

        return super()._train(graph=graph, epochs=epochs, **datasets)