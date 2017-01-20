import tensorflow as tf
import numpy as np
import dataset.utils as utils

class HockeyNetwork:
    def __init__(self, model_file=None):
        self.model_file = model_file
    
    def _train(self, graph, epochs, train_dataset, test_dataset):
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

        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            if self.model_file and utils.path_exists(self.model_file + ".index"):
                print("Loading variables from", self.model_file)
                saver.restore(sess, self.model_file)
            else:
                sess.run(tf.global_variables_initializer())
            
            losses, accuracies = [], []
            for epoch in range(epochs):
                print("Epoch {0}.".format(epoch + 1))

                def train_func(step, x, y, names):
                    feed_dict = {graph['x']: x, graph['y']: y}
                    loss, accuracy, prediction, _ = sess.run([graph['loss'], graph['accuracy'], graph['prediction'], graph['train_step']], feed_dict=feed_dict)
                    
                    print("Train: step - {0:2}, loss - {1:.5f}, accuracy - {2:.5f}, names - {3}, prediction - {4}, y - {5}".format(step, loss, accuracy, np.unique(names), prediction, y), end="\r", flush=True)
                    return loss, accuracy
                
                def train_total(total_loss, total_acc, steps):
                    print(" " * 60, end="\r")
                    print("Train: loss - {0:.5f}, accuracy - {1:.5f}".format(total_loss/steps, total_acc/steps), flush=True)

                result = run(dataset=train_dataset(), step_func=train_func, total_func=train_total)
                losses.extend(result[0])
                accuracies.extend(result[1])

                if self.model_file:
                    model_path, _ = utils.split_full_path(self.model_file)
                    utils.make_sure_path_exists(model_path)
                    print("Save variables to", self.model_file)
                    saver.save(sess, self.model_file)
                
                def test_func(step, x, y, names):
                    feed_dict = {graph['x']: x, graph['y']: y}
                    loss, accuracy, prediction = sess.run([graph['loss'], graph['accuracy'], graph['prediction']], feed_dict=feed_dict)

                    print("Test: step - {0:2}, loss - {1:.5f}, accuracy - {2:.5f}, name - {3}, prediction - {4}, y - {5}".format(step, loss, accuracy, np.unique(names), prediction, y), end="\r", flush=True)
                    return loss, accuracy
                
                def test_total(total_loss, total_acc, steps):
                    print(" " * 60, end="\r")
                    print("Test: loss - {0:.5f}, accuracy - {1:.5f}".format(total_loss/steps, total_acc/steps), flush=True)

                run(dataset=test_dataset(), step_func=test_func, total_func=test_total)

            return losses, accuracies

    def _datasets(self, names_func, train_func, test_func):
        share = dict(names=None)
        def train_dataset():
            train_names, test_names = names_func()
            share['names'] = test_names
            return train_func(train_names)
        
        def test_dataset():
            return test_func(share['names'])
        
        return dict(train_dataset=train_dataset, test_dataset=test_dataset)