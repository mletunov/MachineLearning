import numpy as np
import tensorflow as tf
import os.path

def gen_data(dataset, num_steps, batch_size, frac=0.9):
    def gen_train(names):
        batch_x, batch_y = [], []
        for name in names:
            data_x, data_y = dataset.read_tuple(name)
            for shift in range(len(data_x) - num_steps):
                batch_x.append(data_x[shift:shift + num_steps])
                batch_y.append(data_y)
                if len(batch_x) >= batch_size:
                    yield (np.array(batch_x), np.array(batch_y))
                    batch_x, batch_y = [], []
        if len(batch_x) > 0:
            yield (np.array(batch_x), np.array(batch_y))
    
    def gen_test(names):
        batch_x, batch_y = [], []
        for name in names:
            data_x, data_y = dataset.read_tuple(name)
            for shift in range(len(data_x) - num_steps):
                batch_x.append(data_x[shift:shift + num_steps])
                batch_y.append(data_y)
            yield (name, np.array(batch_x), np.array(batch_y))
    
    train_names, test_names = dataset.split(dataset.read_names(), frac)
    return gen_train(train_names), gen_test(test_names)

def build(num_classes = 2, state_size = 100, num_steps = 30):

    tf.reset_default_graph()
    tf.set_random_seed(100)

    with tf.name_scope("input"):
        # num_steps sequence of frames: height x width x depth
        x = tf.placeholder(tf.float32, (None, num_steps, 288, 360, 3))
        # video class - fight (1) or not (0)
        y = tf.placeholder(tf.int32, (None,))

        flatten_x = tf.reshape(x, (-1, num_steps, np.prod(x.get_shape().as_list()[2:])))

    with tf.name_scope("rnn"):
        rnn_inputs = flatten_x
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, num_classes)
        init_state = cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

    with tf.name_scope("accuracy"):
        score = rnn_outputs[:,-1,:]
        prediction = tf.cast(tf.arg_max(score, dimension=1), tf.int32)        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), tf.float32))
        
    with tf.name_scope("prediction"):
        prediction_one_hot = tf.one_hot(prediction, depth=num_classes)
        prediction_total = tf.reduce_mean(prediction_one_hot, axis=0)

    with tf.name_scope("train"):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(score, y)
        total_loss = tf.reduce_mean(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(total_loss)

    return dict(
            num_steps=num_steps,
            x=x,
            y=y,
            prediction = prediction_total,
            pr = prediction,
            accuracy=accuracy,
            loss=total_loss,
            train_step=train_step)

def train(network, dataset, epochs, batch_size=20, check_file=None):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        if check_file and os.path.exists(check_file):
            print("Loading variables from", check_file)
            saver.restore(sess, check_file)
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print("Epoch {0}.".format(epoch + 1))
                        
            train_step = 0            
            train_data, test_data = gen_data(dataset, network['num_steps'], batch_size, frac=0.9)

            train_loss, train_accuracy = 0, 0
            for idx, (x, y) in enumerate(train_data):
                step = idx + 1

                feed_dict = {network['x']: x, network['y']: y}
                loss, accuracy, prediction, pr, _ = sess.run([network['loss'], network['accuracy'], network['prediction'], network['pr'], network['train_step']], feed_dict=feed_dict)
                train_loss += loss
                train_accuracy += accuracy
                
                print("\r", "Step {0:2}: loss - {1:.5f}, accuracy - {2:.5f}, fight_prob - {3:.2f}, pr - {4}, y - {5}".format(step, loss, accuracy, prediction[1], pr, y), end="")

            print("\r", "Train  : loss - {0:.5f}, accuracy - {1:.5f}".format(train_loss/step, train_accuracy/step))
            
            if check_file:
                print("Save variables to", check_file)
                saver.save(sess, check_file)

            test_loss, test_accuracy = 0, 0
            for idx, (name, x, y) in enumerate(test_data):
                step = idx + 1

                feed_dict = {network['x']: x, network['y']: y}
                loss, accuracy, prediction, pr = sess.run([network['loss'], network['accuracy'], network['prediction'], network['pr']], feed_dict=feed_dict)
                test_loss += loss
                test_accuracy += accuracy
                
                print("\r", "Step {0:2}: loss - {1:.5f}, accuracy - {2:.5f}, name - {3}, fight_prob - {4:.2f}, pr - {5}, y - {6}".format(step, loss, accuracy, name, prediction[1], pr, y), end="")
            
            print("\r", "Test   : loss - {0:.5f}, accuracy - {1:.5f}".format(test_loss/step, test_accuracy/step))