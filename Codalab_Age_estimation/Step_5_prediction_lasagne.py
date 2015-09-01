__author__ = 'dwu'
import gzip
import itertools
import pickle
import os
import sys
import numpy as np
sys.path.append('/idiap/home/dwu/lasagne/lasagne')

import lasagne
import theano
import theano.tensor as T
import time
import cv2

from NN.Lasagne_Age_Regression import _shared, load, build_model

BATCH_SIZE = 64

def create_test_functions(x_, output_layer, X_tensor_type=T.matrix):

    X_batch = X_tensor_type('x')
    predAge = output_layer.get_output(X_batch, deterministic=True)
    iter_valid = theano.function(
        [], predAge,
        givens={
            X_batch: x_,
        },
    )

    return iter_valid

class DataLoaderTest():
    def __init__(self, X, batch_size=64, shuffle=False):
        self.X = X
        self.batch_size = batch_size
        self.n_iter_test = int(np.ceil((self.X.shape[0])/float(batch_size)))
        self.shuffle = shuffle
        self.pos_test = list(np.asarray(range(self.n_iter_test))*self.batch_size)

    def next_test_batch(self, x_):
        pos = self.pos_test.pop(0)
        if len(self.pos_test) >= 1:
            Xb = self.X[pos:pos+self.batch_size]
        elif len(self.pos_test) == 0:
            # the last element batch
            Xb = self.X[pos:]
        Xb = np.squeeze(Xb.astype(np.float32))
        x_.set_value(Xb.swapaxes(1, 3), borrow=True)

def main():
    print("Loading data...")
    load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'
    fname = load_path+"data_with_label_DPM_Color_NoCROP.hdf5"
    X, y, y_mean, y_std, y_train_variance, X_mean, X_std = load(fname, test=True)
    dataloader = DataLoaderTest(X)



    print("Building model and compiling functions...")
    output_layer = build_model(input_width=96, input_height=96, output_dim=1,
                batch_size=BATCH_SIZE, dimshuffle=True)
    #output_layer = build_model()
    print("Loading the trained parameters")
    params_all = pickle.load(open('Lasagne_age_regressor_best.pkl', 'rb'))
    lasagne.layers.set_all_param_values(output_layer, params_all)

    x_ = _shared(np.empty((BATCH_SIZE,3,96,96)))

    iter_funcs = create_test_functions(x_, output_layer, X_tensor_type=T.tensor4)

    now = time.time()
    predAges = np.empty(shape=(dataloader.X.shape[0],1))

    valid_image_list = '/idiap/user/dwu/spyder/Codalab_Age_estimation/valid_matlab_squence.csv'
    Valid_image_name = []
    with open(valid_image_list, 'r') as valid_list_file:
        for image_name in valid_list_file:
            #print image_name[:-1]
            Valid_image_name.append(image_name[:-1])

    image_count = 0
    print "start predicting"
    for b in range(dataloader.n_iter_test):
        dataloader.next_test_batch(x_)
        predAge_batch = iter_funcs()
        #print predAge_batch.shape
        predAges[b*BATCH_SIZE:b*BATCH_SIZE+predAge_batch.shape[0],:] = predAge_batch * y_std + y_mean

        if True:
            #let's save all the faces
            valid_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Validation'
            valid_save_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Valid_crop'
            font = cv2.FONT_HERSHEY_SIMPLEX
            for index in range(len(predAge_batch)):
                color_image = cv2.imread(os.path.join(valid_image_dir, Valid_image_name[image_count+index]))
                cv2.putText(color_image, str(predAge_batch[index] * y_std + y_mean), (0,50), font, 1, (255,0,255))
                cv2.imwrite(os.path.join(valid_save_dir, Valid_image_name[image_count+index]), color_image)
            image_count += BATCH_SIZE

    y_dict = {}
    y_dict['y_pred'] = predAges
    with open(load_path+'valid_y_pred.pkl','wb') as handle:
        pickle.dump(y_dict, handle)
    print("Finish predicting {} images, using time {:.3f}s".format(dataloader.X.shape[0], time.time()-now))






if __name__ == '__main__':
    main()