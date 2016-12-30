import tensorflow as tf

class LaunchNetwork:
    def __init__(self, model_file, graph):
        self.graph = graph
        self.model_file = model_file
    
    def predict(self, x):
        with tf.Session() as sess:
            self.graph()            
            saver = tf.train.Saver()
            saver.restore(self.model_file)
            
            
            
