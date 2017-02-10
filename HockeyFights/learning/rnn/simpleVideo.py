import numpy as np
import tensorflow as tf
import dataset.utils as utils
from learning import baseNetwork

class SimpleModel(baseNetwork.BaseModel):
    def __init__(self, frame, checkpoint_dir=None, seed=None):
        self.frame = frame
        self.seed = seed
        return super().__init__(checkpoint_dir)

    def build(self, rnn_state=100, num_steps=30, avg_result=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.num_classes = 2
            self.num_steps=num_steps

            if self.seed:
                tf.set_random_seed(self.seed)

            with tf.name_scope("input"):
                # batch x epoch x height x width x channels; epoch length = num_steps so we can use multiple epochs in one batch
                x = tf.placeholder(tf.float32, (None, num_steps, *self.frame))
                flatten_x = tf.reshape(x, (-1, num_steps, np.prod(x.get_shape().as_list()[2:])))

            with tf.name_scope("rnn"):
                rnn_inputs = flatten_x
                cell = tf.nn.rnn_cell.GRUCell(rnn_state)
                init_state = cell.zero_state(tf.shape(flatten_x)[0], dtype=tf.float32)
                rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
                output = tf.reshape(rnn_outputs, shape=(-1, rnn_state)) if avg_result else rnn_outputs[:,-1,:]

            with tf.name_scope("dense"):
                dense_w = tf.Variable(super().norm([rnn_state, self.num_classes]), tf.float32, name="w")
                dense_b = tf.Variable(tf.zeros([self.num_classes]), tf.float32, name = "b")
                dense = tf.add(tf.matmul(output, dense_w), dense_b, name="dense")

            with tf.name_scope("prediction"):
                score = tf.reduce_mean(tf.reshape(dense, shape=(-1, num_steps, self.num_classes)), axis = 1) if avg_result else dense
                prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)

            self.input = x
            self.score = score
            self.prediction = prediction
            self.saver = tf.train.Saver(max_to_keep=5)
            self.init = tf.global_variables_initializer()
        return self

class SimpleTrainer(baseNetwork.BaseTrainer):
    def __init__(self, model, **kwargs):
        return super().__init__(model, **kwargs)

    def build(self, learning_rate=1e-2):
        with self.model.graph.as_default():
            with tf.name_scope("input"):
                # video class - fight (1) or not (0)
                y = tf.placeholder(tf.int32, (None,))

            with tf.name_scope("accuracy"):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(self.model.prediction, y), tf.float32), name="accuracy")
                tf.summary.scalar("accuracy", accuracy)

            with tf.name_scope("train"):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.model.score, y)
                total_loss = tf.reduce_mean(loss, name="loss")
                tf.summary.scalar("loss", total_loss)
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

            self.expected = y
            self.train_step = train_step
            self.loss = total_loss
            self.accuracy = accuracy
            self.init = [var.initializer for var in tf.global_variables() if var.name.startswith('train')]
        return self

    def train(self, dataset, epochs, batch_size=20):
        if self.model.seed:
            np.random.seed(self.model.seed)

        # split train and test datasets constantly during whole training 9:1
        train_names, test_names = utils.split(dataset.read_names(), frac = 0.9)

        # shuffle training dataset before each epoch
        train_dataset = lambda: dataset.gen_dataset(utils.shuffle(train_names), by_video=False, frames_count=self.model.num_steps, batch_size=batch_size)

        # it doesn't matter to shuffle test dataset because it will not take any impact on result
        test_dataset = lambda: dataset.gen_dataset(test_names, by_video=True, frames_count=self.model.num_steps, batch_size=batch_size)

        return super()._train(epochs, train_dataset, test_dataset)