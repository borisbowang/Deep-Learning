import pickle
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


# load the dataset
data_file = open("youtube_train_data.pkl", "rb")
train_data, train_labels = pickle.load(data_file)
data_file.close()

res_train = train_data.astype('uint8')

# plt.imshow(res_train[1, 2, :, :, :])
# plt.scatter(x=train_labels[1, 2, :, 0], y=train_labels[1, 2, :, 1], c='r', s=50, marker='x')
# plt.show()

train_data = train_data.astype(np.float32)
train_labels = train_labels.astype(np.float32)

# normalize the training data
train_data -= np.mean(train_data, axis=(2, 3, 4), keepdims=True)
train_data /= np.std(train_data, axis=(2, 3, 4), keepdims=True)


# divide into training & validation
# valid_ind = np.random.randint(0, 8000, 700)
# valid_x = train_data[valid_ind, :, :, :, :]
# valid_y = train_labels[valid_ind, :, :, :]
# train_x = np.delete(train_data, valid_ind, axis=0)
# train_y = np.delete(train_labels, valid_ind, axis=0)
train_x = train_data[:7500, :, :, :, :]
train_y = train_labels[:7500, :, :, :]
valid_x = train_data[7500:, :, :, :, :]
valid_y = train_labels[7500:, :, :, :]


# input
x = tf.placeholder(shape=[None, 10, 64, 64, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10, 7, 2], dtype=tf.float32)

batch_size = 5
h_units = 128

# weights & bias
Weights = {'W_c1': tf.get_variable('W_c1', shape=[5, 5, 3, 32],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_c2': tf.get_variable('W_c2', shape=[5, 5, 32, 64],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_c3': tf.get_variable('W_c3', shape=[3, 3, 64, 128],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_fc': tf.get_variable('W_fc', shape=[h_units, 14],
                                    initializer=tf.contrib.layers.xavier_initializer())}

Bias = {'b_c1': tf.Variable(tf.zeros(shape=[32], dtype=tf.float32)),
        'b_c2': tf.Variable(tf.zeros(shape=[64], dtype=tf.float32)),
        'b_c3': tf.Variable(tf.zeros(shape=[128], dtype=tf.float32)),
        'b_fc': tf.Variable(tf.zeros(shape=[14], dtype=tf.float32))}


# CNN ###################################
# cnn_inp = tf.reshape(x, [batch_size*10, 64, 64, 3])
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
cnn_fc = tf.reshape(pool3, [-1, 8 * 8 * 128])

rnn_inp = tf.reshape(cnn_fc, [-1, 10, 8 * 8 * 128])


# RNN #####################################
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=h_units)

h_val, _ = tf.nn.dynamic_rnn(lstm_cell, rnn_inp, dtype=tf.float32)

# final_output = tf.zeros(shape=[batch_size, 0, 14])
# for i in np.arange(10):
#     temp = tf.reshape(h_val[:, i, :], [batch_size, h_units])
#     output = tf.matmul(temp, Weights['W_fc']) + Bias['b_fc']
#     output = tf.reshape(output, [-1, 1, 14])
#     final_output = tf.concat([final_output, output], axis=1)

temp = tf.reshape(h_val, [-1, h_units])
output = tf.matmul(temp, Weights['W_fc']) + Bias['b_fc']
final_output = tf.reshape(output, [-1, 10, 14])


# loss function and optimizer
y_true = tf.reshape(y, [-1, 10, 14])
loss_mse = tf.reduce_mean(tf.squared_difference(y_true, final_output))
optimizer = tf.train.AdamOptimizer().minimize(loss_mse)


# prediction operation
predict_op = tf.reshape(final_output, [-1, 10, 7, 2])

# pixel distance error
sq_diff = tf.squared_difference(predict_op, y)
p_dist = tf.sqrt(tf.reduce_sum(sq_diff, axis=3))
avg_dist = tf.reduce_mean(p_dist)
# avg_dist = tf.reduce_sum(p_dist)

# prediction accuracy within 20 pix
pd_seq = np.arange(start=0.0, stop=20.0, step=0.1)
n = len(pd_seq)


def p_acc(pix_dist, pd_seq, n):
    pix_dist = np.reshape(pix_dist, [-1, 7])
    # M, _ = tf.shape(pix_dist)
    acc = np.zeros(shape=[7, n], dtype=np.float32)
    for i in range(7):
        for j in range(n):
            acc[i, j] = np.mean(np.less_equal(pix_dist[:, i], pd_seq[j]).astype(np.float32))
    return acc


# Create the collection.
tf.get_collection("validation_nodes")

# Add stuff to the collection.
tf.add_to_collection("validation_nodes", x)
tf.add_to_collection("validation_nodes", y)
tf.add_to_collection("validation_nodes", predict_op)


# start training
saver = tf.train.Saver()

avg_d_tr = []
avg_d_tt = []
loss_tr = []
loss_tt = []

N_epochs = 2000
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)

    for i in range(N_epochs):
        batch_ind = np.random.randint(0, 7500, batch_size)
        batch_x = train_x[batch_ind, :, :, :, :]
        batch_y = train_y[batch_ind, :, :, :]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        print(i)

        if ((i+1) % 20) == 0:
            l_tr, d_tr = sess.run([loss_mse, avg_dist], feed_dict={x: batch_x, y: batch_y})
            v_batch_ind = np.random.randint(0, 500, 10)
            v_batch_x = valid_x[v_batch_ind, :, :, :, :]
            v_batch_y = valid_y[v_batch_ind, :, :, :]
            l_tt, d_tt = sess.run([loss_mse, avg_dist], feed_dict={x: v_batch_x, y: v_batch_y})
            avg_d_tr.append(d_tr)
            avg_d_tt.append(d_tt)
            loss_tr.append(l_tr)
            loss_tt.append(l_tt)
            print(l_tr, d_tr, l_tt, d_tt)

    # accuracy for each joint
    v_batch_ind = np.random.randint(0, 500, 100)
    v_batch_x = valid_x[v_batch_ind, :, :, :, :]
    v_batch_y = valid_y[v_batch_ind, :, :, :]

    y_hat, p_d, loss_final, avg_d_final = sess.run([predict_op, p_dist, loss_mse, avg_dist],
                                                   feed_dict={x: v_batch_x, y: v_batch_y})
    acc = p_acc(p_d, pd_seq, n)

    # plot the 1st image from 1st sequence and true/predicted joints
    old_ind = range(7500, 8000, 1)[v_batch_ind[0]]

    plt.imshow(res_train[old_ind, 0, :, :, :])
    plt.scatter(x=train_labels[old_ind, 0, :, 0], y=train_labels[old_ind, 0, :, 1], c='r', s=50, marker='x')
    plt.scatter(x=y_hat[0, 0, :, 0], y=y_hat[0, 0, :, 1], c='b', s=50, marker='x')
    plt.show()

    # save the trained model
    save_path = saver.save(sess, "./my_model")


# final loss & average pixel distance on 100 validation data (sequences)
print(loss_final, avg_d_final)


# plot acc
for i in range(7):
    plt.plot(pd_seq, acc[i, :])
plt.legend(['head',
            'right shoulder',
            'left shoulder',
            'right wrist',
            'left wrist',
            'right elbow',
            'left elbow'], loc='upper left')
plt.xlabel('pixel distance from GT')
plt.ylabel('accuracy')
plt.show()


# Learning Curve
# plot average distance of training & test vs. iteration
plt.plot(range(19, 2000, 20), avg_d_tr)
plt.plot(range(19, 2000, 20), avg_d_tt)
plt.legend(['training pixel distance', 'test pixel distance'], loc='upper right')
plt.xlabel('training iterations')
plt.show()

# plot average loss of training & test vs. iteration
plt.plot(range(19, 2000, 20), loss_tr)
plt.plot(range(19, 2000, 20), loss_tt)
plt.legend(['training loss', 'test loss'], loc='upper right')
plt.xlabel('training iterations')
plt.show()



