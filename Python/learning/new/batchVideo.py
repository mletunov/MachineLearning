import numpy as np
import tensorflow as tf
import dataset.utils as utils
from learning.baseVideo import HockeyNetwork

class BatchVideoNetwork():
    def __init__(self, checkpoint_dir=None, seed=None):
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        return super().__init__()

    def build(self, rnn_state=100, num_steps=30):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.num_classes = 2
            self.num_steps=num_steps
            frame_size = (288, 360, 3)
       
            if self.seed:
                tf.set_random_seed(self.seed)

            with tf.name_scope("input"):               
                # batch x epoch x height x width x channels; epoch length = num_steps so we can use multiple epochs in one batch
                x = tf.placeholder(tf.float32, (None, num_steps, *frame_size))
                flatten_x = tf.reshape(x, (-1, num_steps, np.prod(x.get_shape().as_list()[2:])))
        
            with tf.name_scope("rnn"):
                rnn_inputs = flatten_x
                cell = tf.nn.rnn_cell.GRUCell(rnn_state)
                init_state = cell.zero_state(tf.shape(flatten_x)[0], dtype=tf.float32)
                rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
                output = rnn_outputs[:,-1,:]
    
            with tf.name_scope("dense"):
                dense_w = tf.Variable(tf.truncated_normal([rnn_state, self.num_classes]), tf.float32, name="w")
                dense_b = tf.Variable(tf.zeros([self.num_classes]), tf.float32, name = "b")
                dense = tf.add(tf.matmul(output, dense_w), dense_b, name="dense")
        
            with tf.name_scope("prediction"):
                score = dense
                prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)
            
            self.input = x
            self.score = score
            self.prediction = prediction
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()        
        return self
    def save(self, sess, checkpoint_dir=None, global_step=None):
        if not checkpoint_dir:
            checkpoint_dir = self.checkpoint_dir
        
        if not checkpoint_dir:
            raise ValueError("checkpoint directory is not specified")
        
        utils.make_sure_path_exists(checkpoint_dir)
        path = self.saver.save(sess, utils.path_join(checkpoint_dir, 'model'), global_step=global_step)
        print("Saved graph with variables to", path)
        return path
    def load(self, sess, checkpoint_dir=None):
        storage = self.checkpoint_dir
        if storage and utils.path_exists(storage):
            checkpoint = tf.train.get_checkpoint_state(storage)
            if checkpoint and len(checkpoint.all_model_checkpoint_paths) > 0:
                self.saver.recover_last_checkpoints(checkpoint.all_model_checkpoint_paths)
                print("Loading variables from", self.saver.last_checkpoints[-1])
                self.saver.restore(sess, self.saver.last_checkpoints[-1])
                return
        
        print("Initializing variables randomly")
        sess.run(self.init)

    def execute(self, func):
        with tf.Session(graph=self.graph) as sess:
            self.load(sess)
            return func(sess)
    def predict(self, x, sess=None):
        def predict_impl(sess):
            feed_dict={self.input: x if hasattr(x, '__len__') else [x]}
            prediction = sess.run(self.prediction, feed_dict=feed_dict)
            return prediction

        return self.execute(predict_impl) if sess is None else predict_impl(sess)

class BatchVideoTrainer():
    def __init__(self, model):
        self.model = model
    
    def build(self, learning_rate=1e-2):
        with self.model.graph.as_default():
            with tf.name_scope("input"):
                # video class - fight (1) or not (0)
                y = tf.placeholder(tf.int32, (None,))

            #with tf.name_scope("prediction"):
            #    prediction_one_hot = tf.one_hot(self.model.prediction, depth=self.model.num_classes)
            #    prediction_total = tf.reduce_mean(prediction_one_hot, axis=0)

            with tf.name_scope("accuracy"):       
                accuracy = tf.reduce_mean(tf.cast(tf.equal(self.model.prediction, y), tf.float32))
        
            with tf.name_scope("train"):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.model.score, y)
                total_loss = tf.reduce_mean(loss)
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
            
            self.expected = y
            self.train_step = train_step
            self.loss = total_loss
            self.accuracy = accuracy
            self.init = [var.initializer for var in tf.global_variables() if var.name.startswith('train')]
        return self
    def train(self, dataset, epochs, batch_size=20):
        def counter(run):
            def wrapper(*args, **kwargs):
                result = list(zip(*run(*args,**kwargs)))  
                count_steps = np.amin([20, (len(result) // np.amin([3, len(result)]))])
                
                total_loss, total_accuracy = 0.0, 0.0
                count_loss, count_accuracy = 0.0, 0.0
                losses, accuracies = [], []
                for idx, (loss, accuracy) in enumerate(result):
                    total_loss += loss
                    count_loss += loss
                    total_accuracy += accuracy
                    count_accuracy += accuracy
                    
                    if (idx + 1) % count_steps == 0:
                        losses.append(count_loss/count_steps)
                        accuracies.append(count_accuracy/count_steps)
                        count_loss, count_accuracy = 0.0, 0.0
                
                total_func = kwargs['total_func']
                total_func(total_loss, total_accuracy, len(result))
                return losses, accuracies
            return wrapper

        @counter
        def run(dataset, step_func, total_func):
            losses, accuracies = [], []
            for idx, (x, y, names) in enumerate(dataset):
                loss, accuracy = step_func(idx + 1, x, y, names)

                losses.append(loss)
                accuracies.append(accuracy)
            
            return losses, accuracies
        
        if self.model.seed:
            np.random.seed(self.model.seed)

        train_dataset, test_dataset = self._datasets(
                        lambda: utils.split(dataset.read_names(), frac = 0.5), 
                        lambda train_names: dataset.gen_dataset(train_names, by_video=False, frames_count=self.model.num_steps, batch_size=batch_size),
                        lambda test_names: dataset.gen_dataset(test_names, by_video=True, frames_count=self.model.num_steps, batch_size=batch_size))
        
        
        def train_function(sess):
            sess.run(self.init)
            
            losses, accuracies = [], []
            for epoch in range(epochs):
                print("Epoch {0}.".format(epoch + 1))

                def train_func(step, x, y, names):
                    feed_dict = {self.model.input: x, self.expected: y}
                    loss, accuracy, prediction, _ = sess.run([self.loss, self.accuracy, self.model.prediction, self.train_step], feed_dict=feed_dict)
                    
                    print("Train: step - {0:2}, loss - {1:.5f}, accuracy - {2:.5f}, names - {3}, prediction - {4}, y - {5}".format(step, loss, accuracy, np.unique(names), prediction, y), end="\r", flush=True)
                    return loss, accuracy
                
                def train_total(total_loss, total_acc, steps):
                    print(" " * 60, end="\r")
                    print("Train: loss - {0:.5f}, accuracy - {1:.5f}".format(total_loss/steps, total_acc/steps), flush=True)

                result = run(dataset=train_dataset(), step_func=train_func, total_func=train_total)
                losses.extend(result[0])
                accuracies.extend(result[1])
                
                self.model.save(sess, global_step=(epoch + 1))

                def test_func(step, x, y, names):
                    feed_dict = {self.model.input: x, self.expected: y}
                    loss, accuracy, prediction = sess.run([self.loss, self.accuracy, self.model.prediction], feed_dict=feed_dict)

                    print("Test: step - {0:2}, loss - {1:.5f}, accuracy - {2:.5f}, name - {3}, prediction - {4}, y - {5}".format(step, loss, accuracy, np.unique(names), prediction, y), end="\r", flush=True)
                    return loss, accuracy
                
                def test_total(total_loss, total_acc, steps):
                    print(" " * 60, end="\r")
                    print("Test: loss - {0:.5f}, accuracy - {1:.5f}".format(total_loss/steps, total_acc/steps), flush=True)

                run(dataset=test_dataset(), step_func=test_func, total_func=test_total)

            return losses, accuracies

        return self.model.execute(train_function)

    def _datasets(self, names_func, train_func, test_func):
        share = dict(names=None)
        def train_dataset():
            train_names, test_names = names_func()
            share['names'] = test_names
            return train_func(train_names)
        
        def test_dataset():
            return test_func(share['names'])
        
        return train_dataset, test_dataset
