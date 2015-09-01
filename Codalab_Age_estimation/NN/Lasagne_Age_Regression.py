__author__ = 'dwu'

import itertools
import sys
import numpy as np
sys.path.append('/idiap/home/dwu/lasagne/lasagne')
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import cuda_convnet
from sklearn.utils import shuffle



def plot_flip_face(X, X_mean, X_std):
    X = X*X_std+X_mean
    import cv2
    from matplotlib import pyplot
    Xf = X[:, :, :, ::-1, :]
    fig = pyplot.figure(figsize=(6,3))
    ax1 = fig.add_subplot(1,2,1, xticks=[], yticks=[])
    ax2 = fig.add_subplot(1,2,2, xticks=[], yticks=[])
    for framenum in range(10):
        img1 = X[framenum,0]
        b,g,r = cv2.split(img1)
        img1 = cv2.merge([r,g,b])
        img2 = Xf[framenum,0]
        b,g,r = cv2.split(img2)
        img2 = cv2.merge([r,g,b])
        ax1.imshow(img1)
        ax2.imshow(img2)
        pyplot.show()
        cv2.waitKey(2000)

def _shared(val, borrow=True):
    return theano.shared(np.array(val, dtype=theano.config.floatX), borrow=borrow)

def load(fname, test=False):
    import h5py
    # load pandas dataframe
    f = h5py.File(fname, "r")
    x_train_image_croped = f["x_train_image_croped"][:]
    X_mean = x_train_image_croped.mean(axis=0)
    X_std = x_train_image_croped.std(axis=0)

    x_valid_image_croped = f["x_valid_image_croped"][:]
    y_train_age = f["y_train_age"][:]
    y_train_variance = f["y_train_variance"][:]
    y_mean = y_train_age.mean()
    y_std = y_train_age.std()

    # only FTRAIN has any target columns
    if not test:
        X = np.expand_dims(x_train_image_croped,axis=1)
        X = (X - X_mean) / X_std
        X = X.astype(np.float32)
        y = y_train_age
        # scale target coordinates to around [-1, 1]

        y = (y - y_mean) / y_std
        # shuffle train data
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        X= np.expand_dims(x_valid_image_croped,axis=1)
        X = (X - X_mean) / X_std
        X = X.astype(np.float32)
        y = None
    return X, y, y_mean, y_std, y_train_variance, X_mean, X_std

class DataLoader():
    def __init__(self, X, y, validation_num=None, batch_size=64, shuffle=False):
        self.x_train = X[:-validation_num]
        self.y_train = y[:-validation_num]
        self.x_valid = X[-validation_num:]
        self.y_valid = y[-validation_num:]
        self.batch_size = batch_size
        self.n_iter_train = int(np.floor((self.x_train.shape[0])/float(batch_size)))
        self.n_iter_valid = int(np.floor(self.x_valid.shape[0]/float(batch_size)))
        self.shuffle = shuffle
        self.shuffle_train()
        self.shuffle_valid()

    def shuffle_train(self):
        self.pos_train = list(np.random.permutation(self.n_iter_train)*self.batch_size)

    def shuffle_valid(self):
        self.pos_valid = list(np.asarray(range(self.n_iter_valid))*self.batch_size)

    def next_train_batch(self, x_, y_):
        if len(self.pos_train) == 0: self.shuffle_train()
        pos = self.pos_train.pop()
        Xb = self.x_train[pos:pos+self.batch_size]
        yb = self.y_train[pos:pos+self.batch_size]

        indices = np.random.choice(self.batch_size, self.batch_size / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1, :]
        Xb = np.squeeze(Xb.astype(np.float32))
        #yb = np.squeeze(yb.astype(np.float32))
        yb = yb.astype(np.float32)
        #print Xb.swapaxes(1, 3).shape
        x_.set_value(Xb.swapaxes(1, 3), borrow=True)
        y_.set_value(yb, borrow=True)

    def next_valid_batch(self, x_, y_):
        if len(self.pos_valid) == 0: self.shuffle_valid()
        pos = self.pos_valid.pop()
        Xb = self.x_valid[pos:pos+self.batch_size]
        yb = self.y_valid[pos:pos+self.batch_size]

        Xb = np.squeeze(Xb.astype(np.float32))
        #yb = np.squeeze(yb.astype(np.float32))
        yb = yb.astype(np.float32)
        x_.set_value(Xb.swapaxes(1, 3), borrow=True)
        y_.set_value(yb, borrow=True)

def build_model(input_width=96, input_height=96, output_dim=1,
                batch_size=64, dimshuffle=True):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 3, input_width, input_height),
    )

    if not dimshuffle:
        l_in = cuda_convnet.bc01_to_c01b(l_in)

    l_conv1 = cuda_convnet.Conv2DCCLayer(
        l_in,
        num_filters=32,
        filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool1 = cuda_convnet.MaxPool2DCCLayer(
        l_conv1,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    l_conv2 = cuda_convnet.Conv2DCCLayer(
        l_pool1,
        num_filters=64,
        filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool2 = cuda_convnet.MaxPool2DCCLayer(
        l_conv2,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    l_conv3 = cuda_convnet.Conv2DCCLayer(
        l_pool2,
        num_filters=128,
        filter_size=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        dimshuffle=dimshuffle,
    )
    l_pool3 = cuda_convnet.MaxPool2DCCLayer(
        l_conv3,
        pool_size=(2, 2),
        dimshuffle=dimshuffle,
    )

    if not dimshuffle:
        l_poo3 = cuda_convnet.c01b_to_bc01(l_pool3)

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool3,
        num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
    )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_hidden2 = lasagne.layers.DenseLayer(
         l_hidden1_dropout,
         num_units=512,
         nonlinearity=lasagne.nonlinearities.rectify,
         )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.linear,
    )

    return l_out


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def create_iter_functions(x_,
                          y_,
                          output_layer,
                          X_tensor_type=T.matrix,
                          learning_rate=0.01,
                          momentum=0.9):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    X_batch = X_tensor_type('x')
    y_batch = T.fmatrix('y')

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.mse)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [], loss_train,
        updates=updates,
        givens={
            X_batch: x_,
            y_batch: y_,
        },
    )

    iter_valid = theano.function(
        [], loss_eval,
        givens={
            X_batch: x_,
            y_batch: y_,
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
    )


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def train(iter_funcs, dataloader, x_, y_):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataloader.n_iter_train
    num_batches_valid = dataloader.n_iter_valid

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            dataloader.next_train_batch(x_, y_)
            batch_train_loss = iter_funcs['train']()
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        for b in range(num_batches_valid):
            dataloader.next_valid_batch(x_, y_)
            batch_valid_loss = iter_funcs['valid']()
            batch_valid_losses.append(batch_valid_loss)

        avg_valid_loss = np.mean(batch_valid_losses)


        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
        }

