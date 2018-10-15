import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle

# read training label and data
dir_path = os.path.dirname(os.path.realpath(__file__))
filename_ltr = dir_path + "/Prog3_data/labels/train_label.txt"
label_tr = np.loadtxt(filename_ltr).astype(np.int8)

M_tr = len(label_tr)
x_tr = np.zeros([M_tr, 784]).astype(np.float32)
y_tr = np.zeros([M_tr, 10]).astype(np.float32)
for m in range(M_tr):
    filename_cur = dir_path + "/Prog3_data/train_data/" + str((m + 1)).zfill(5) + ".jpg"
    x_m = mpimg.imread(filename_cur).reshape(28 * 28).astype(np.float32)
    x_tr[m, :] = x_m / 255.0
    # change label to 1-of-K encoding
    ind = label_tr[m]
    y_tr[m, ind] = 1


# read test label and data
filename_ltt = dir_path + "/Prog3_data/labels/test_label.txt"
label_tt = np.loadtxt(filename_ltt).astype(np.int8)

M_tt = len(label_tt)
x_tt = np.zeros([M_tt, 784]).astype(np.float32)
y_tt = np.zeros([M_tt, 10]).astype(np.float32)
for m in range(M_tt):
    filename_cur = dir_path + "/Prog3_data/test_data/" + str((m+1)).zfill(5) + ".jpg"
    x_m = mpimg.imread(filename_cur).reshape(28*28).astype(np.float32)
    x_tt[m, :] = x_m / 255.0
    # change label to 1-of-K encoding
    ind = label_tt[m]
    y_tt[m, ind] = 1


# define all the variables
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# initialize weight
W1 = tf.Variable(tf.random_normal([784, 100])*0.01, tf.float32, name="W1")
W10 = tf.Variable(tf.zeros(100), tf.float32, name="W10")

W2 = tf.Variable(tf.random_normal([100, 100])*0.01, tf.float32, name="W1")
W20 = tf.Variable(tf.zeros(100), tf.float32, name="W10")

W3 = tf.Variable(tf.random_normal([100, 10])*0.01, tf.float32, name="W1")
W30 = tf.Variable(tf.zeros(10), tf.float32, name="W10")

# learning rate
eta = tf.constant(0.02, tf.float32, name='eta')


# Sigmoid_M() function
def sig_m(z):
    # sigm = tf.div(tf.exp(-z), tf.reduce_sum(tf.exp(-z), 0))
    sigm = tf.nn.softmax(z)
    return sigm


# Derivative of Sigmoid_M(z)
def dsig_m(z):
    dsigm = tf.multiply(sig_m(z), tf.subtract(1.0, sig_m(z)))
    return dsigm


# Relu(z)
def Relu(z):
    rl = tf.maximum(0.0, z)
    return rl


# Derivative of Relu(z)
def dRule(z):
    drl = tf.sign(tf.nn.relu(z))
    return drl


# Compute classification error
def class_error(yhat, y, n):
    total = np.zeros(10, np.float32)
    err = np.zeros(10, np.float32)
    for j in range(n):
        pred = np.argmax(yhat[j, :])
        true = np.argmax(y[j, :])
        # print(pred, true)
        total[true] = total[true] + 1
        if pred != true:
            err[true] = err[true] + 1
    err_rate = err / total
    avg_err = sum(err) / sum(total)
    avg_loss = np.mean((np.sum(np.square((y-yhat)), axis=1) / 2), 0)
    return err_rate, avg_err, avg_loss


# Forward-propagation
H1 = tf.nn.relu(tf.matmul(x, W1) + W10)
H2 = tf.nn.relu(tf.matmul(H1, W2) + W20)
Yhat = tf.nn.softmax(tf.matmul(H2, W3) + W30)
# # loss function (squared loss)
# loss_val = tf.reduce_mean(tf.div(tf.reduce_sum(tf.square(tf.subtract(y, Yhat)), 1), 2))

# Back-propagation
# Compute gradients
# gradient of Yhat (use squared loss)
dYhat = tf.subtract(Yhat, y)

# Output layer
dz3 = tf.multiply(dYhat, dsig_m(tf.matmul(H2, W3) + W30))
dW3 = tf.matmul(tf.transpose(H2), dz3)
dW30 = tf.reduce_mean(dz3, 0)
dH2 = tf.matmul(dz3, tf.transpose(W3))

# Hidden layer H2
dz2 = tf.multiply(dH2, dRule(tf.matmul(H1, W2) + W20))
dW2 = tf.matmul(tf.transpose(H1), dz2)
dW20 = tf.reduce_mean(dz2, 0)
dH1 = tf.matmul(dz2, tf.transpose(W2))

# Hidden layer H1 (Input layer)
dz1 = tf.multiply(dH1, dRule(tf.matmul(x, W1) + W10))
dW1 = tf.matmul(tf.transpose(x), dz2)
dW10 = tf.reduce_mean(dz1, 0)


# Update Weights Operations
up_W3 = tf.assign(W3, tf.subtract(W3, tf.multiply(dW3, eta)))
up_W30 = tf.assign(W30, tf.subtract(W30, tf.multiply(dW30, eta)))
up_W2 = tf.assign(W2, tf.subtract(W2, tf.multiply(dW2, eta)))
up_W20 = tf.assign(W20, tf.subtract(W20, tf.multiply(dW20, eta)))
up_W1 = tf.assign(W1, tf.subtract(W1, tf.multiply(dW1, eta)))
up_W10 = tf.assign(W10, tf.subtract(W10, tf.multiply(dW10, eta)))


# Iterations
each_err_tr = []
avg_err_tr = []
avg_loss_tr = []
each_err_tt = []
avg_err_tt = []
avg_loss_tt = []

batch_size = 50
model = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(model)

    for i in range(10000):
        batch_ind = np.random.randint(0, M_tr, batch_size)
        batch_x = x_tr[batch_ind, :]
        batch_y = y_tr[batch_ind, :]
        # update weights use batched data
        sess.run([up_W3, up_W30, up_W2, up_W20, up_W1, up_W10], feed_dict={x: batch_x, y: batch_y})

        # compute the training error every 100 iteration
        if ((i+1)%100) == 0:
            yhat_tr = sess.run(Yhat, feed_dict={x: x_tr})
            err1_tr, err2_tr, loss_tr = class_error(yhat_tr, y_tr, M_tr)
            # loss_tr = sess.run(loss_val, feed_dict={x: x_tr, y: y_tr})
            each_err_tr.append(err1_tr)
            avg_err_tr.append(err2_tr)
            avg_loss_tr.append(loss_tr)

            # compute the test error
            yhat_tt = sess.run(Yhat, feed_dict={x: x_tt})
            err1_tt, err2_tt, loss_tt = class_error(yhat_tt, y_tt, M_tt)
            # loss_tt = sess.run(loss_val, feed_dict={x: x_tt, y: y_tt})
            each_err_tt.append(err1_tt)
            avg_err_tt.append(err2_tt)
            avg_loss_tt.append(loss_tt)

            print(err2_tr, loss_tr)

        # save weights in last step into Theta
        if i == 9999:
            Theta = [sess.run(W1), sess.run(W10), sess.run(W2), sess.run(W20), sess.run(W3), sess.run(W30)]

        print(i)


# Results
# training and test error for each digit, and average error after last iteration
print(each_err_tr[99], avg_err_tr[99])
print(each_err_tt[99], avg_err_tt[99])


# plot average error of training & test vs. iteration
plt.plot(range(99, 10000, 100), avg_err_tr)
plt.plot(range(99, 10000, 100), avg_err_tt)
plt.legend(['training error', 'test error'], loc='upper right')
plt.show()

# plot average loss of training & test vs. iteration
plt.plot(range(99, 10000, 100), avg_loss_tr)
plt.plot(range(99, 10000, 100), avg_loss_tt)
plt.legend(['training loss', 'test loss'], loc='upper right')
plt.show()


# save the weights
# check theta and pickle
# print(Theta[0][0:1, 0:5])

filehandler = open("nn_parameters.txt", "wb")
pickle.dump(Theta, filehandler, protocol=2)
filehandler.close()





