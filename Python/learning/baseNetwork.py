import tensorflow as tf
import numpy as np
import dataset.utils as utils

class BaseModel():
    """ base model for Hockey dataset """
    def __init__(self, checkpoint_dir=None, **kwargs):
        self._checkpoint_dir = checkpoint_dir
        utils.make_sure_path_exists(checkpoint_dir)
        return super().__init__(**kwargs)

    @staticmethod
    def norm(shape):
        return tf.random_normal(shape=shape, mean=0.0, stddev=0.05, dtype=tf.float32)
    @staticmethod
    def norm2(shape):
        return tf.truncated_normal(shape=shape, stddev=0.1)

    def save(self, sess, checkpoint_dir=None, global_step=None):
        if not checkpoint_dir:
            checkpoint_dir = self._checkpoint_dir

        if not checkpoint_dir:
            raise ValueError("checkpoint directory is not specified")

        path = self.saver.save(sess, utils.path_join(checkpoint_dir, 'model'), global_step=global_step)
        print("Saved graph with variables to", path)
        return path
    def load(self, sess, checkpoint_dir=None):
        checkpoint_dir = self._checkpoint_dir
        if checkpoint_dir and utils.path_exists(checkpoint_dir):
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoint and len(checkpoint.all_model_checkpoint_paths) > 0:
                self.saver.recover_last_checkpoints(checkpoint.all_model_checkpoint_paths)
                print("Loading variables from", self.saver.last_checkpoints[-1])
                self.saver.restore(sess, self.saver.last_checkpoints[-1])
                return

        print("Initializing variables randomly")
        sess.run(self.init)

    def execute(self, func):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(graph=self.graph, config=config) as sess:
            self.load(sess)
            return func(sess)
    def predict(self, x, sess=None):
        def predict_impl(sess):
            feed_dict={self.input: x if hasattr(x, '__len__') else [x]}
            if hasattr(self, "keep_prob"):
                feed_dict[self.keep_prob] = 1.0
            prediction = sess.run(self.prediction, feed_dict=feed_dict)
            return prediction

        return self.execute(predict_impl) if sess is None else predict_impl(sess)

class BaseTrainer():
    """ base trainer for Hockey model """
    def __init__(self, model, **kwargs):
        self.model = model
        return super().__init__(**kwargs)

    @staticmethod
    def _indicators(loop):
        def wrapper(*args, total_func, **kwargs):
            result = list(zip(*loop(*args, total_func=total_func, **kwargs)))
            count_steps = np.amin([20, (len(result) // np.amin([3, len(result)]))])

            total_loss, total_accuracy = 0.0, 0.0
            count_loss, count_accuracy = 0.0, 0.0
            losses, accuracies, summaries = [], [], []
            for idx, (loss, accuracy, summary) in enumerate(result):
                total_loss += loss
                count_loss += loss
                total_accuracy += accuracy
                count_accuracy += accuracy

                if (idx + 1) % count_steps == 0:
                    losses.append(count_loss/count_steps)
                    accuracies.append(count_accuracy/count_steps)
                    summaries.append(summary)
                    count_loss, count_accuracy = 0.0, 0.0

            total_func(total_loss, total_accuracy, len(result))
            return losses, accuracies, summaries
        return wrapper

    @staticmethod
    def _loop(dataset, step_func, total_func):
        losses, accuracies, summaries = [], [], []
        for idx, (x, y, names) in enumerate(dataset):
            loss, accuracy, summary = step_func(idx + 1, x, y, names)

            losses.append(loss)
            accuracies.append(accuracy)
            summaries.append(summary)

        return losses, accuracies, summaries

    def _train(self, epochs, train_dataset, test_dataset, dropout=0.75):
        def train_function(sess):
            sess.run(self.init)

            losses, accuracies = [], []
            indicator_loop = BaseTrainer._indicators(BaseTrainer._loop)
            all_summary = tf.summary.merge_all()
            train_logger = tf.summary.FileWriter(utils.path_join(self.model._checkpoint_dir, "train"), sess.graph)
            test_logger = tf.summary.FileWriter(utils.path_join(self.model._checkpoint_dir, "test"), sess.graph)

            @utils.timeit
            def epoch_func(epoch):
                print("Epoch {0}.".format(epoch + 1))

                # train
                @utils.timeit
                def train_func(step, x, y, names):
                    feed_dict = {self.model.input: x, self.expected: y}
                    if hasattr(self.model, "keep_prob"):
                        feed_dict[self.model.keep_prob] = dropout

                    loss, accuracy, prediction, summary, _ = sess.run([self.loss, self.accuracy, self.model.prediction, all_summary, self.train_step], feed_dict=feed_dict)

                    print("Train step - {0:2}: loss - {1:.5f}, accuracy - {2:.5f}, names - {3}, prediction - {4}, y - {5}".format(step, loss, accuracy, np.unique(names), prediction, y), flush=True)
                    return loss, accuracy, summary

                def train_total(total_loss, total_acc, steps):
                    print("----- Train: loss - {0:.5f}, accuracy - {1:.5f}".format(total_loss/steps, total_acc/steps), flush=True)

                train_losses, train_accuracies, train_summaries = indicator_loop(dataset=train_dataset(), step_func=train_func, total_func=train_total)
                losses.extend(train_losses)
                accuracies.extend(train_accuracies)

                last_step = int(str.split(self.model.saver.last_checkpoints[-1], '-')[-1]) if len(self.model.saver.last_checkpoints) > 0 else 0
                self.model.save(sess, global_step=last_step + 1)
                for idx, summary in enumerate(train_summaries):
                    train_logger.add_summary(summary, int((last_step + (idx + 1) / len(train_summaries)) * 10))

                # test
                @utils.timeit
                def test_func(step, x, y, names):
                    feed_dict = {self.model.input: x, self.expected: y}
                    if hasattr(self.model, "keep_prob"):
                        feed_dict[self.model.keep_prob] = 1.0

                    loss, accuracy, prediction, summary = sess.run([self.loss, self.accuracy, self.model.prediction, all_summary], feed_dict=feed_dict)

                    print("Test step - {0:2}: loss - {1:.5f}, accuracy - {2:.5f}, name - {3}, prediction - {4}, y - {5}".format(step, loss, accuracy, np.unique(names), prediction, y), flush=True)
                    return loss, accuracy, summary

                def test_total(total_loss, total_acc, steps):
                    print("----- Test: loss - {0:.5f}, accuracy - {1:.5f}".format(total_loss/steps, total_acc/steps), flush=True)

                test_losses, test_accuracies, test_summaries = indicator_loop(dataset=test_dataset(), step_func=test_func, total_func=test_total)
                for idx, summary in enumerate(test_summaries):
                    test_logger.add_summary(summary, int((last_step + (idx + 1) / len(test_summaries)) * 10))

            for epoch in range(epochs):
                epoch_func(epoch)

            return losses, accuracies

        return self.model.execute(train_function)

class Predictor:
    def __init__(self, model, **kwargs):
        self._model = model
        return super().__init__(**kwargs)

    def predict(self, dataset, batch_size = 20):
        names = dataset.read_names()
        gen_dataset = lambda: dataset.gen_dataset(names, by_video=True, frames_count=self._model.num_steps, batch_size=batch_size)

        def predict_func(sess):
            result = {}
            prediction = []
            current = None

            def add_result(name, prediction, expected):
                score = np.mean(prediction)
                result[name] = {'score': score, 'expected': expected, 'prediction': 1 if score >= 0.5 else 0}
                result[name]['text'] = "fight" if result[name]['prediction'] == 1 else "no fight"
                print("{3:2} {0}: score - {1:.5f}, prediction - {2}".format(name, result[name]['score'], result[name]['text'], len(result)))

            for x, y, names in gen_dataset():
                if current and names[0] != current:
                    add_result(current, prediction, expected)
                    prediction = []

                current = names[0]
                prediction.extend(self._model.predict(x, sess))
                expected = y[0]

            add_result(current, prediction, expected)
            return result

        return self._model.execute(predict_func)

    def accuracy(self, prediction):
         return np.mean([1 if v['expected'] == v['prediction'] else 0 for _, v in prediction.items()])