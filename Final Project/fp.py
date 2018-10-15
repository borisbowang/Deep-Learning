import pickle
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import seaborn as sn
import pandas as pd


# load the dataset
data_file_p1 = open("youtube_action_train_data_part1.pkl", "rb")
train_data_p1, train_labels_p1 = pickle.load(data_file_p1)
data_file_p1.close()

data_file_p2 = open("youtube_action_train_data_part2.pkl", "rb")
train_data_p2, train_labels_p2 = pickle.load(data_file_p2)
data_file_p2.close()

train_data = np.concatenate((train_data_p1, train_data_p2), axis=0)
train_labels = np.concatenate((train_labels_p1, train_labels_p2), axis=0)

# res_train = train_data.astype('uint8')
#
# plt.imshow(res_train[1, 2, :, :, :])
# plt.show()

# normalize the training data
train_data = train_data.astype(np.float32) / 255.0
train_data -= np.mean(train_data, axis=(2, 3, 4), keepdims=True)
train_data /= np.std(train_data, axis=(2, 3, 4), keepdims=True)

train_labels = train_labels.astype(np.int64)


# divide into training & validation
sf_index = np.arange(7272)
np.random.seed(seed=1)
np.random.shuffle(sf_index)
valid_ind = sf_index[:500]
train_ind = sf_index[500:]
valid_x = train_data[valid_ind, :, :, :, :]
valid_y = train_labels[valid_ind]
# train_x = train_data[train_ind, :, :, :, :]
# train_y = train_labels[train_ind]


# input
x = tf.placeholder(shape=[None, 30, 64, 64, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int64)

# hidden nodes in LSTM
h_units = 128

# weights & bias
# Weights = {'W_c1': tf.Variable(tf.random_normal(shape=[5, 5, 3, 32], stddev=1/128)),
#            'W_c2': tf.Variable(tf.random_normal(shape=[5, 5, 32, 64], stddev=1/128)),
#            'W_c3': tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=1/128)),
#            'W_fc': tf.Variable(tf.random_normal(shape=[h_units, 11], stddev=1/h_units))}

Weights = {'W_c1': tf.get_variable('W_c1', shape=[5, 5, 3, 32],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_c2': tf.get_variable('W_c2', shape=[5, 5, 32, 32],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_c3': tf.get_variable('W_c3', shape=[3, 3, 64, 64],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_fc': tf.get_variable('W_fc', shape=[h_units, 11],
                                    initializer=tf.contrib.layers.xavier_initializer())}

Bias = {'b_c1': tf.Variable(tf.zeros(shape=[32], dtype=tf.float32)),
        'b_c2': tf.Variable(tf.zeros(shape=[32], dtype=tf.float32)),
        'b_c3': tf.Variable(tf.zeros(shape=[64], dtype=tf.float32)),
        'b_fc': tf.Variable(tf.zeros(shape=[11], dtype=tf.float32))}


# CNN ###################################
cnn_inp = tf.reshape(x, [-1, 64, 64, 3])
# Convolutional Layer 1
conv1 = tf.nn.conv2d(cnn_inp, Weights['W_c1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, Bias['b_c1'])
conv1 = tf.nn.relu(conv1)
# Pooling Layer 1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional Layer 2
conv2 = tf.nn.conv2d(pool1, Weights['W_c2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, Bias['b_c2'])
conv2 = tf.nn.relu(conv2)
# Pooling Layer 2
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional Layer 3
conv3 = tf.nn.conv2d(pool2, Weights['W_c3'], strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, Bias['b_c3'])
conv3 = tf.nn.relu(conv3)
# Pooling Layer 3
pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Dense Layer (which is the RNN input)
cnn_fc = tf.reshape(pool3, [-1, 8 * 8 * 64])

rnn_inp = tf.reshape(cnn_fc, [-1, 30, 8 * 8 * 64])


# # Normalize rnn_inp
# rnn_inp = tf.subtract(rnn_inp, tf.reduce_mean(rnn_inp, axis=(0, 1), keepdims=True))
# rnn_inp = tf.nn.l2_normalize(rnn_inp, axis=(0, 1))


# RNN #####################################
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_units)

h_val, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_inp, dtype=tf.float32)

temp = tf.reshape(h_val[:, -1, :], [-1, h_units])
output = tf.matmul(temp, Weights['W_fc']) + Bias['b_fc']
y_hat = tf.nn.softmax(output)


# loss function and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)


# prediction operation
predict_op = tf.argmax(y_hat, 1)

# overall accuracy
correct = tf.equal(predict_op, y)
acc_all = tf.reduce_mean(tf.cast(correct, 'float'))
# confusion matrix
confusion = tf.confusion_matrix(labels=y, predictions=predict_op, num_classes=11)


# Compute classification error for each class
def class_error(predict_y, true_y, n):
    total = np.zeros(11, np.float32)
    err = np.zeros(11, np.float32)
    for j in range(n):
        pred = predict_y[j]
        true = true_y[j]
        total[true] = total[true] + 1
        if pred != true:
            err[true] = err[true] + 1
    err_rate = err / total
    return err_rate


# Create the collection.
tf.get_collection("validation_nodes")

# Add stuff to the collection.
tf.add_to_collection("validation_nodes", x)
tf.add_to_collection("validation_nodes", y)
tf.add_to_collection("validation_nodes", predict_op)


# start training
saver = tf.train.Saver()

batch_size_tr = 128
batch_size_ts = 100

acc_tr = []
acc_ts = []
cost_tr = []
cost_ts = []

N_epochs = 1000
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)

    for i in range(N_epochs):
        batch_ind = np.random.choice(np.arange(6727), batch_size_tr, replace=False)
        batch_x = train_data[train_ind[batch_ind], :, :, :, :]
        batch_y = train_labels[train_ind[batch_ind]]

        # cnn_fc1, rnn_inp1 = sess.run([cnn_fc, rnn_inp], feed_dict={x: batch_x, y: batch_y})

        sess.run([optimizer], feed_dict={x: batch_x, y: batch_y})
        print(i)

        if ((i+1) % 20) == 0:
            c_tr, ac_tr = sess.run([cost, acc_all], feed_dict={x: batch_x, y: batch_y})

            v_batch_ind = np.random.choice(np.arange(500), batch_size_ts, replace=False)
            v_batch_x = train_data[valid_ind[v_batch_ind], :, :, :, :]
            v_batch_y = train_labels[valid_ind[v_batch_ind]]

            c_ts, ac_ts = sess.run([cost, acc_all], feed_dict={x: v_batch_x, y: v_batch_y})

            acc_tr.append(ac_tr)
            acc_ts.append(ac_ts)
            cost_tr.append(c_tr)
            cost_ts.append(c_ts)

            print(c_tr, ac_tr, c_ts, ac_ts)

    # save the trained model
    save_path = saver.save(sess, "./my_model")

    # error rate for each class
    pred_y, ac_final, con_mx = sess.run([predict_op, acc_all, confusion], feed_dict={x: valid_x, y: valid_y})
    err_ts = class_error(pred_y, valid_y, 500)
    print(ac_final)

# Results
# final error rate for each class
print(err_ts)

# confusion matrix
print(con_mx)

# plot average error of training & test vs. iteration
plt.plot(range(19, 1000, 20), acc_tr)
plt.plot(range(19, 1000, 20), acc_ts)
plt.legend(['training accuracy', 'test accuracy'], loc='upper right')
plt.show()

# plot average loss of training & test vs. iteration
plt.plot(range(19, 1000, 20), cost_tr)
plt.plot(range(19, 1000, 20), cost_ts)
plt.legend(['training loss', 'test loss'], loc='upper right')
plt.show()

# plot confusion matrix
names = ["b_shooting", "cycling", "diving", "g_swinging", "h_riding", "s_juggling",
        "swinging", "t_swinging", "t_jumping", "v_spiking", "walking"]

df_cm = pd.DataFrame(con_mx, names, names)
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="Blues")  # font size

