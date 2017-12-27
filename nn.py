# nn.py
import tensorflow as tf
import numpy as np
from time import time


def train_and_predict(X_train, y_train, X_test):

    # define width of each layer
    layer_width = {
        'fc1': 1000,
        'fc2': 600,
        'fc3': 300,
        'out': y_train.shape[1]
    }

    # fixed parameters
    n_classes = len(np.unique(y_train))
    init_std = 0.03

    # weights and biases
    weights = {
        'fc1': tf.Variable(tf.truncated_normal([X_train.shape[1],layer_width['fc1']], 
                                               stddev=init_std), trainable=True),
        'fc2': tf.Variable(tf.truncated_normal([layer_width['fc1'],layer_width['fc2']], 
                                               stddev=init_std), trainable=True),
        'fc3': tf.Variable(tf.truncated_normal([layer_width['fc2'],layer_width['fc3']], 
                                               stddev=init_std), trainable=True),
        'out': tf.Variable(tf.truncated_normal([layer_width['fc3'],layer_width['out']], 
                                               stddev=init_std), trainable=True)
    }

    biases = {
        'fc1': tf.Variable(tf.truncated_normal([layer_width['fc1']], 
                                               stddev=init_std), trainable=True),
        'fc2': tf.Variable(tf.truncated_normal([layer_width['fc2']], 
                                               stddev=init_std), trainable=True),
        'fc3': tf.Variable(tf.truncated_normal([layer_width['fc3']], 
                                               stddev=init_std), trainable=True),
        'out': tf.Variable(tf.truncated_normal([layer_width['out']], 
                                               stddev=init_std), trainable=True)
    }

    # create neural net
    def neural_net(x, weights, biases):
        
        # lay1
        fc1 = tf.add(tf.matmul(x, weights['fc1']), biases['fc1'])
        fc1 = tf.nn.relu(fc1)

        # lay1
        fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])
        fc2 = tf.tanh(fc2)

        # lay1
        fc3 = tf.add(tf.matmul(fc2, weights['fc3']), biases['fc3'])
        fc3 = tf.tanh(fc3)

        # lay1
        out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
        
        return out

    # learning parameters
    batch_size = 100000
    training_epochs = 2

    # graph input
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]))
    y = tf.placeholder(tf.int32, shape=(None))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.04, global_step, decay_steps=X_train.shape[0]/batch_size, 
                                               decay_rate=.80, staircase=True)
    logits = neural_net(x, weights, biases)

    # loss, optimizer, and variables initialization 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, y, name='xentropy')
    reg = .01
    loss = (tf.reduce_mean(cross_entropy) + 
            reg * tf.nn.l2_loss(weights['fc1']) + 
            reg * tf.nn.l2_loss(weights['fc2']) + 
            reg * tf.nn.l2_loss(weights['fc3']) + 
            reg * tf.nn.l2_loss(weights['out'])) 
            
    # optimizer
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    #evaluation function
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(logits,1 ))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #init
    init = tf.initialize_all_variables()

    # launch graph
    sess = tf.Session()
    sess.run(init)

    t0 = time()
    # create session variables
    feed_dict={
        x: None,
        y: None
    }

    num_samples = 50000

    # sample train and validation set
    np.random.seed(42)
    indices_train = np.random.permutation(y_train.shape[0])[:num_samples]

    # init scores
    scores_train = []
    losses = []
    epochs = []

    # training cycle
    for epoch in range(1, training_epochs+1):

        t1 = time()
        # set size of batch
        total_batch = int(X_train.shape[0]/batch_size)+1

        # loop over batches
        for i in range(total_batch):                
            feed_dict[x] = X_train[i*batch_size:(i+1)*batch_size]
            feed_dict[y] = y_train[i*batch_size:(i+1)*batch_size]
            _, loss_value = sess.run([train_op, loss], feed_dict)

        # update last loss value
        losses.append(loss_value)

        print("Epoch: {:0>4}, Cost: {:.8f}, Time: {:.2f}".format(epoch, losses[-1], time()-t1)) 

    print("Optimization Finished! Time to complete: {:.2f}".format(time()-t0))

    # save model to restore later
    save_file = 'train_model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, save_file)

    softmax_pred = tf.nn.top_k(tf.nn.softmax(logits), 7)
    feed_dict[x] = X_test
    classes = sess.run(softmax_pred, feed_dict) 
    return classes
