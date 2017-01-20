import tensorflow as tf
import numpy as np
import dataset.utils as utils
from azure.storage.blob import BlockBlobService

class Model:
    def __init__(self, storage=None, **kwargs):
        self.storage = storage
        return super().__init__(**kwargs)

    def build(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, shape=(None), name="input")
            self.x = self.input
            self.W = tf.Variable(tf.random_normal(shape=(), dtype=tf.float32), name="W") 
            self.b = tf.Variable(tf.zeros(shape=(), dtype=tf.float32), name="b")
            self.output = tf.add(tf.mul(self.W, self.x), self.b, name="output")
            self.saver = tf.train.Saver()

            self.init = tf.global_variables_initializer()

        return self
    
    def execute(self, func):
        with tf.Session(graph=self.graph) as sess:
            self.load(sess)            
            return func(sess)

    def save(self, sess, storage=None, global_step=None):
        if not storage:
            storage = self.storage
        
        if not storage:
            raise ValueError("storage s not specified")
        
        utils.make_sure_path_exists(storage)
        path = self.saver.save(sess, utils.path_join(storage, 'model'), global_step=global_step)
        print("Saved graph with variables to", path)
        return path

    def load(self, sess, storage=None):
        storage = self.storage if not storage else storage
        if storage and utils.path_exists(storage):
            checkpoint = tf.train.get_checkpoint_state(storage)
            if checkpoint and len(checkpoint.all_model_checkpoint_paths) > 0:
                self.saver.recover_last_checkpoints(checkpoint.all_model_checkpoint_paths)
                print("Loading variables from", self.saver.last_checkpoints[-1])
                self.saver.restore(sess, self.saver.last_checkpoints[-1])
                return
        
        print("Initializing variables randomly")
        sess.run(self.init)

    def predict(self, x, sess=None):
        def predict_impl(sess):
            feed_dict={self.input: x if hasattr(x, '__len__') else [x]}
            output = sess.run(self.output, feed_dict=feed_dict)
            return output if len(feed_dict[self.input]) > 1 else output[0]

        return self.execute(predict_impl) if sess is None else predict_impl(sess)

class Trainer:
    def __init__(self, model):
        self.model = model

    def build(self, learning_rate=1e-2):
        with self.model.graph.as_default():
            self.expected = tf.placeholder(tf.float32, shape=(None), name="exp")
            self.loss = tf.nn.l2_loss(self.model.output - self.expected, name="loss")
            with tf.name_scope("train"):
                self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
                self.init = [var.initializer for var in tf.global_variables() if var.name.startswith('train')]
            
        return self
        
    def train(self, x, y, epochs, storage=None):        
        def train_func(sess):
            sess.run(self.init)
            for i in range(epochs):
                feed_dict = {self.model.x: x, self.expected: y} 
                loss, _ = sess.run([self.loss, self.train_step], feed_dict=feed_dict)
                if (i + 1) % 100 == 0:
                    print("{0}: loss - {1}".format(i + 1, loss))                    
                    self.model.save(sess, global_step=(i + 1)//100)
        
        self.model.execute(train_func)

class AzureModel(Model):
    def __init__(self, account, key, storage, **kwargs):
        self.service = BlockBlobService(account_name=account, account_key=key)
        return super().__init__(storage, **kwargs)
        
    def save(self, sess, storage = None, global_step = None):
        path = super().save(sess, storage, global_step)
        right_path = path.replace('\\', '/')
        self.service.create_blob_from_path(container_name='models',blob_name=right_path, file_path=right_path + ".meta")
        
model = AzureModel(account='datastora', key='z7+emBG4BJD0xnI9v4apLaa4mxCScFUudNyVfJQn53b3XC4FozjXZ85wC3wHRTFC/TYOosssoxZcrFYR2ClfIA==', storage='tmp/data').build()

model.execute(lambda sess: (
    print(model.predict(2.0, sess)),
    print(model.predict(3.0, sess)),
    print(model.predict(3.0, sess))))

print(model.predict(4.0))

trainer = Trainer(model).build()
trainer.train(x=[2.0, 3.0, 4.0], y=[8.1, 11.5, 15.7], epochs=1000)

model.execute(lambda sess: (
    print(model.predict(2.0, sess)),
    print(model.predict(3.0, sess)),
    print(model.predict(3.0, sess))))

print(model.predict(4.0))