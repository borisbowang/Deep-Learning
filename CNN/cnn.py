from __future__ import print_function
import pickle
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
import os

from io import BytesIO
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML


data_file = open("cifar_10_tf_train_test.pkl", "rb")
train_x, train_y, test_x, test_y = pickle.load(data_file, encoding='latin1')
data_file.close()

# cv2.imshow("test", test_x[1])
# cv2.waitKey()

# Preprocessing
train_x = train_x.astype(np.float32)
train_mean = np.mean(train_x, axis=0)
train_x = (train_x - train_mean) / 255

test_x = test_x.astype(np.float32)
test_mean = np.mean(test_x, axis=0)
test_x = (test_x - test_mean) / 255

train_y = np.asarray(train_y, dtype=np.int64)
test_y = np.asarray(test_y, dtype=np.int64)

# input
x = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
y = tf.placeholder(shape=[None], dtype=tf.int64)
t_input = tf.placeholder(np.float32, name='input')

# weights & bias
# Weights = {'W_conv1': tf.Variable(tf.random_normal(shape=[5, 5, 3, 32], stddev=1/64)),
#            'W_conv2': tf.Variable(tf.random_normal(shape=[5, 5, 32, 32], stddev=1/32)),
#            'W_conv3': tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=1/32)),
#            'W_fc': tf.Variable(tf.random_normal(shape=[8*8*64, 10], stddev=1/(8*8*64)))}

Weights = {'W_conv1': tf.get_variable('W_conv1', shape=[5, 5, 3, 32],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_conv2': tf.get_variable('W_conv2', shape=[5, 5, 32, 32],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_conv3': tf.get_variable('W_conv3', shape=[3, 3, 32, 64],
                                      initializer=tf.contrib.layers.xavier_initializer()),
           'W_fc': tf.get_variable('W_fc', shape=[8*8*64, 10],
                                    initializer=tf.contrib.layers.xavier_initializer())}

Bias = {'b_conv1': tf.Variable(tf.zeros(shape=[32], dtype=tf.float32)),
        'b_conv2': tf.Variable(tf.zeros(shape=[32], dtype=tf.float32)),
        'b_conv3': tf.Variable(tf.zeros(shape=[64], dtype=tf.float32)),
        'b_fc': tf.Variable(tf.zeros(shape=[10], dtype=tf.float32))}


# Convolutional Layer 1
conv1 = tf.nn.conv2d(x, Weights['W_conv1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, Bias['b_conv1'])
conv1 = tf.nn.relu(conv1)
# Pooling Layer 1
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional Layer 2
conv2 = tf.nn.conv2d(pool1, Weights['W_conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, Bias['b_conv2'])
conv2 = tf.nn.relu(conv2)
# Pooling Layer 2
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional Layer 3
conv3 = tf.nn.conv2d(pool2, Weights['W_conv3'], strides=[1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.bias_add(conv3, Bias['b_conv3'])
conv3 = tf.nn.relu(conv3)

# Dense Layer
fc = tf.reshape(conv3, [-1, 8 * 8 * 64])
out = tf.nn.bias_add(tf.matmul(fc, Weights['W_fc']), Bias['b_fc'])
y_hat = tf.nn.softmax(out)


# cost and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.01
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step)
# learning_rate=0.001


predict_op = tf.argmax(y_hat, 1)
correct = tf.equal(predict_op, y)
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


# Compute classification error for each class
def class_error(predict_y, y, n):
    total = np.zeros(10, np.float32)
    err = np.zeros(10, np.float32)
    for j in range(n):
        pred = predict_y[j]
        true = y[j]
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


# CNN training
saver = tf.train.Saver()

avg_acc_tr = []
avg_acc_tt = []
cost_tr = []
cost_tt = []

N_epochs = 15000
batch_size = 64
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)

    for i in range(N_epochs):
        batch_ind = np.random.randint(0, 50000, batch_size)
        batch_x = train_x[batch_ind, :, :, :]
        batch_y = train_y[batch_ind]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if ((i+1) % 100) == 0:
            c_tr, acc_tr = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            c_tt, acc_tt = sess.run([cost, accuracy], feed_dict={x: test_x, y: test_y})
            avg_acc_tr.append(acc_tr)
            avg_acc_tt.append(acc_tt)
            cost_tr.append(c_tr)
            cost_tt.append(c_tt)
            print(i, c_tr, acc_tr, c_tt, acc_tt)
    # save the trained model
    save_path = saver.save(sess, "/Users/borisbowang/PycharmProjects/DL3/my_model")

    # error rate for each class
    pred_y = sess.run(predict_op, feed_dict={x: test_x})
    err_tt = class_error(pred_y, test_y, 5000)
    print(err_tt)


#     def showarray(a, fmt='jpeg'):
#         a = np.uint8(np.clip(a, 0, 1) * 255)
#         f = BytesIO()
#         PIL.Image.fromarray(a).save(f, fmt)
#         display(Image(data=f.getvalue()))
#
#
#     def visstd(a, s=0.1):
#         '''Normalize the image range for visualization'''
#         return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5
#
#
# # def T(layer):
# #     '''Helper for getting layer output tensor'''
# #     return graph.get_tensor_by_name("import/%s:0" % layer)
#
#
#     def render_naive(W, n, iter_n=20, step=1.0):
#         img0 = np.random.uniform(size=(1, 32, 32, 3)) + 100.0
#         t_obj = tf.nn.conv2d(t_input, W, strides=[1, 1, 1, 1], padding="VALID")
#         t_obj = tf.nn.relu(t_obj)
#         t_score = tf.reduce_mean(t_obj[:, :, :, n])  # defining the optimization objective
#         t_grad = tf.gradients(t_score, t_input)[0]  # behold the power of automatic differentiation!
#         img = img0.copy()
#         for i in range(iter_n):
#             g, score = sess.run([t_grad, t_score], feed_dict={t_input: img})
#             # normalizing the gradient, so the same step size should work
#             g /= g.std() + 1e-8  # for different layers and networks
#             img += g * step
#         print(np.uint8(np.clip(visstd(img), 0, 1) * 255))
#         showarray(visstd(img))
#
#     for j in range(32):
#         render_naive(Weights['W_conv1'], j)


def visualize(w1, b1, nth):
    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(1, 32, 32, 3)) + 100.0
    first_layers = conv(t_input, w1, b1)
    layer = first_layers[:, :, :, nth]
    t_score = tf.reduce_mean(layer)
    t_grad = tf.gradients(t_score, t_input)[0]
    img = img_noise.copy()
    iter_n = 100
    step = 1.0
    for i in range(iter_n):
        score, g = sess.run([t_score, t_grad], feed_dict={t_input: img})
        # normalizing the gradient, so the same step size should work
        g /= g.std() + 1e-8  # for different layers and networks
        img += g * step
    showarray(visstd(img[0, :, :, :]), nth)


def conv(x, W, b):
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    # b=tf.tile(tf.expand_dims(b,0),[x.get_shape().as_list()[0],1,1,1])
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def showarray(a, nth, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1) * 255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    im = PIL.Image.fromarray(a)
    display(Image(data=f.getvalue()))
    im.save('/Users/borisbowang/PycharmProjects/DL3/' + str(nth) + '.jpeg')


def visstd(a, s=0.1):
    return (a - a.mean()) / max(a.std(), 1e-4) * s + 0.5

b1 = np.zeros((32))
for i in range(32):
    visualize(Weights['W_conv1'], b1, i)


# plot average error of training & test vs. iteration
plt.plot(range(99, 15000, 100), avg_acc_tr)
plt.plot(range(99, 15000, 100), avg_acc_tt)
plt.legend(['training accuracy', 'test accuracy'], loc='upper right')
plt.show()

# plot average loss of training & test vs. iteration
plt.plot(range(99, 15000, 100), cost_tr)
plt.plot(range(99, 15000, 100), cost_tt)
plt.legend(['training loss', 'test loss'], loc='upper right')
plt.show()






