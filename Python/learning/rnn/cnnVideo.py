import numpy as np
import tensorflow as tf
import dataset.utils as utils
from learning.rnn import baseNetwork

class CnnModel(baseNetwork.BaseModel):
    def __init__(self, checkpoint_dir=None, seed=None):
        self.seed = seed
        return super().__init__(checkpoint_dir)

    def build(self, rnn_state=20, num_steps=30, avg_result=False, batch_norm=False, dropout=False):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.num_classes = 2
            self.num_steps=num_steps
            frame_size = (288, 360, 3)

            if self.seed:
                tf.set_random_seed(self.seed)

            with tf.name_scope("input"):
                # num_steps sequence of frames: height x width x depth
                x = tf.placeholder(tf.float32, (None, num_steps, *frame_size))
                if dropout:
                    self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

                flatten_x = tf.reshape(x, (-1, *frame_size))

            with tf.name_scope("cnn"):
                with tf.name_scope("conv1"):
                    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 16), stddev=0.1), name="w")
                    conv1_b = tf.Variable(tf.zeros(shape=(16)), name="b")
                    conv1 = tf.nn.relu(tf.nn.conv2d(flatten_x, conv1_w, strides=[1, 2, 2, 1], padding="VALID") + conv1_b, name="conv")
                
                if batch_norm:
                    with tf.name_scope("batch_norm"):
                        batch_mean, batch_var = tf.nn.moments(conv1,[0])
                        scale = tf.Variable(tf.ones(tf.shape(batch_mean)))
                        beta = tf.Variable(tf.zeros(tf.shape(batch_mean)))
                        # Small epsilon value for the BN transform
                        epsilon = 1e-3
                        conv1 = tf.nn.batch_normalization(conv1, batch_mean, batch_var, beta, scale, epsilon)

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

            if dropout:
                flatten_conv = tf.nn.dropout(flatten_conv, self.keep_prob, name="dropout")

            with tf.name_scope("rnn"):
                rnn_inputs = tf.reshape(flatten_conv, (-1, num_steps, size))
                cell = tf.nn.rnn_cell.GRUCell(rnn_state)
                init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
                rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
                output = tf.reshape(rnn_outputs, shape=(-1, rnn_state)) if avg_result else rnn_outputs[:,-1,:]

                with tf.name_scope("dense"):
                    dense_w = tf.Variable(tf.truncated_normal([rnn_state, self.num_classes]), tf.float32, name="w")
                    dense_b = tf.Variable(tf.zeros([self.num_classes]), tf.float32, name = "b")
                    dense = tf.add(tf.matmul(output, dense_w), dense_b, name="out")

            with tf.name_scope("prediction"):
                score = tf.reduce_mean(tf.reshape(dense, shape=(-1, num_steps, self.num_classes)), axis = 1) if avg_result else dense
                prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)

            self.input = x
            self.score = score
            self.prediction = prediction
            self.saver = tf.train.Saver(max_to_keep=5)
            self.init = tf.global_variables_initializer()
        return self

class CnnTrainer(baseNetwork.BaseTrainer):
    def __init__(self, model, **kwargs):
        return super().__init__(model, **kwargs)

    def build(self, learning_rate=1e-2, l2_loss=False):
        with self.model.graph.as_default():
            with tf.name_scope("input"):
                # video class - fight (1) or not (0)
                y = tf.placeholder(tf.int32, (None,))

            with tf.name_scope("accuracy"):
                accuracy = tf.reduce_mean(tf.cast(tf.equal(self.model.prediction, y), tf.float32), name="accuracy")
                tf.summary.scalar("accuracy", accuracy)

            with tf.name_scope("train"):
                if l2_loss:
                    loss = tf.nn.l2_loss(self.model.score - tf.one_hot(y, depth=self.model.num_classes))
                else:
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.model.score, y)

                total_loss = tf.reduce_mean(loss)
                tf.summary.scalar("loss", total_loss)
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

            self.expected = y
            self.train_step = train_step
            self.loss = total_loss
            self.accuracy = accuracy
            self.init = [var.initializer for var in tf.global_variables() if var.name.startswith('train')]
        return self

    def train(self, dataset, epochs, batch_size=20, dropout=0.75):
        if self.model.seed:
            np.random.seed(self.model.seed)

        train_names, test_names = utils.split(dataset.read_names(), frac = 0.9)
        train_dataset = lambda: dataset.gen_dataset(train_names, by_video=False, frames_count=self.model.num_steps, batch_size=batch_size)
        test_dataset = lambda: dataset.gen_dataset(test_names, by_video=True, frames_count=self.model.num_steps, batch_size=batch_size)

        return super()._train(epochs, train_dataset, test_dataset, dropout)