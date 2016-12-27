import tensorflow as tf
import numpy as np

def gen_data(dataset):
    names = dataset.read_names()
    for name in names:
        yield dataset.read_tuple(name)

def build(num_classes = 2, state_size = 100):

    tf.reset_default_graph()
    tf.set_random_seed(100)

    with tf.name_scope("input"):
        # sequence of frames: height x width x depth
        x = tf.placeholder(tf.float32, (None, 288, 360, 3))
        # video class - fight (1) or not (0)
        y = tf.placeholder(tf.int32, ())

        flatten_x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

    with tf.name_scope("rnn"):
        rnn_inputs = tf.expand_dims(flatten_x, axis=0)
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, num_classes)
        init_state = cell.zero_state(1, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
        output = rnn_outputs[:,-1,:]

    with tf.name_scope("accuracy"):
        score = tf.squeeze(output, axis=[0])
        prediction = tf.cast(tf.arg_max(score, dimension=0), tf.int32)        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))

    with tf.name_scope("train"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(score, y)
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    return dict(
            x=x,
            y=y,
            accuracy=accuracy,
            loss=loss,
            train_step=train_step)

def train(network, dataset, epochs):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print("Epoch {0}.".format(epoch + 1))
            
            total_loss, total_accuracy = 0, 0
            for idx, (x, y) in enumerate(gen_data(dataset)):
                feed_dict = {network['x']: x, network['y']: y}
                loss, accuracy, _ = sess.run([network['loss'], network['accuracy'], network['train_step']], feed_dict=feed_dict)
                total_loss += loss
                total_accuracy += accuracy
                step = idx + 1
                print("\r", "Step {0}: loss - {1}, accuracy - {2}".format(step, loss, accuracy), end="")
            print("\r", "loss - {0}, accuracy - {1}".format(total_loss/step, total_accuracy/step))