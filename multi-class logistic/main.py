import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle

# setattr(tf.contrib.rnn.GRUCell, '__deepcopy__', lambda self, _: self)
# setattr(tf.contrib.rnn.BasicLSTMCell, '__deepcopy__', lambda self, _: self)
# setattr(tf.contrib.rnn.MultiRNNCell, '__deepcopy__', lambda self, _: self)

# read training label and data
dir_path = os.path.dirname(os.path.realpath(__file__))
filename_ltr = dir_path + "/data_prog/labels/train_label.txt"
label_tr = np.loadtxt(filename_ltr)

M = len(label_tr)

x_tr = np.zeros([M, 785]).astype(np.float32)
y_tr = np.zeros([M, 5]).astype(np.float32)
for m in range(M):
    filename_cur = dir_path + "/data_prog/train_data/" + str((m+1)).zfill(5) + ".jpg"
    x_m = mpimg.imread(filename_cur).reshape(28*28).astype(np.float32)
    # append 1
    x_m = np.append(x_m/255, [1])
    x_tr[m,:]=x_m
    # change label to 1-of-K encoding
    ind = int(label_tr[m]) - 1
    y_tr[m, ind] = 1


# read test label and data
filename_ltest = dir_path + "/data_prog/labels/test_label.txt"
label_test = np.loadtxt(filename_ltest)

M_test = len(label_test)

x_test = np.zeros([M_test, 785]).astype(np.float32)
y_test = np.zeros([M_test, 5]).astype(np.float32)
for m in range(M_test):
    filename_cur = dir_path + "/data_prog/test_data/" + str((m+1)).zfill(4) + ".jpg"
    x_m = mpimg.imread(filename_cur).reshape(28*28).astype(np.float32)
    # append 1
    x_m = np.append(x_m/255, [1])
    x_test[m, :] = x_m
    # change label to 1-of-K encoding
    ind = int(label_test[m]) - 1
    y_test[m, ind] = 1


# compute the gradient for W
def Gradient(W, X, y, lamb,n):
    res = tf.matmul(X, W)
    res_exp = tf.exp(res)
    # compute sigmoid_m function
    sig_M = tf.transpose(tf.transpose(res_exp)/tf.reduce_sum(res_exp, axis=1))
    gradient = -1/n*tf.matmul(tf.transpose(X), tf.subtract(y, sig_M)) + 2*lamb*W
    return gradient


# compute classification error
def ClassifyError(yhat, y, n):
    total = [0]*5
    err = [0]*5
    err_rate = [0.0]*5
    for j in range(n):
        pred = np.argmax(yhat[j, :])
        #print(yhat[j, :],pred)
        true = np.argmax(y[j, :])
        if pred == true:
            total[true] = total[true] + 1
        else:
            total[true] = total[true] + 1
            err[true] = err[true] + 1
    for j in range(5):
        err_rate[j] = err[j]/total[j]
    avg_err = sum(err)/sum(total)
    err_rate.append(avg_err)
    return err_rate


# train the weights W using training data, through SGD
# initialization
W = tf.random_normal([785, 5])*0.01
batch_size = 100
eta = 0.7
lamb = 0.002
# SGD
err_tr = np.zeros([150, 6])
err_test = np.zeros([150, 6])
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(150):
    batch_ind = np.random.randint(0, M/batch_size)
    batch_x = x_tr[(batch_ind*batch_size):((batch_ind+1)*batch_size), :]
    batch_y = y_tr[(batch_ind*batch_size):((batch_ind+1)*batch_size), :]
    g = Gradient(W, batch_x, batch_y, lamb, batch_size)
    W = tf.subtract(W, eta*g)
    # print(sess.run(W[1,]))
    # training error
    yhat = tf.nn.softmax(tf.matmul(batch_x, W))
    # print(sess.run(yhat[1,]))
    err_tr[i, :] = ClassifyError(sess.run(yhat), batch_y, batch_size)
    # testing error
    yhat_test = tf.nn.softmax(tf.matmul(x_test, W))
    err_test[i, :] = ClassifyError(sess.run(yhat_test), y_test, M_test)

    print(i)


# plot the training error & test error vs. iteration
plt.plot(range(150), err_tr[:, 5])
plt.plot(range(150), err_test[:, 5])
plt.legend(['training error', 'test error'], loc='upper right')

plt.show()

# yhat_test = tf.nn.softmax(tf.matmul(x_test, W))
# err_test = ClassifyError(sess.run(yhat_test), y_test, M_test)
print(err_test[149, :])

# plot the weights
W = W.eval(session=sess)
# print(W[0:2, :])

for i in range(5):
    img = W[0:784, i].reshape(28, 28)
    plt.imshow(img)
    plt.colorbar()
    plt.show()


filehandler = open("multiclass_parameters.txt", "wb")
pickle.dump(W, filehandler)
filehandler.close()


# objects = []
# with (open("multiclass_parameters.txt", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break

# print(objects[0][0:2, :])



