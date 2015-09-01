#!/usr/bin/env python
__author__ = 'dwu'

import sys
import numpy as np
import cPickle as pickle
import time

sys.path.append('/idiap/home/dwu/lasagne/lasagne')
from NN.Lasagne_Age_Regression import _shared, load, DataLoader, build_model, create_iter_functions, train
import lasagne
import theano.tensor as T

NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
VALIDATIONNUM = BATCH_SIZE * 4

def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'
    fname = load_path+"data_with_label_DPM_Color_NoCROP.hdf5"
    X, y, y_mean, y_std, y_train_variance, X_mean, X_std = load(fname)
    dataloader = DataLoader(X, y, validation_num=VALIDATIONNUM, batch_size=BATCH_SIZE)

    if True:
        print("Building model and compiling functions...")
        output_layer = build_model()
        x_ = _shared(np.empty((BATCH_SIZE,3,96,96)))
        y_ = _shared(np.empty((BATCH_SIZE,1)))

        iter_funcs = create_iter_functions(
            x_, y_,
            output_layer,
            X_tensor_type=T.tensor4,
            learning_rate=LEARNING_RATE
        )

    if True:
        print("Loading the trained parameters")
        params_all = pickle.load(open('Lasagne_age_regressor_E200.pkl', 'rb'))
        lasagne.layers.set_all_param_values(output_layer, params_all)

    print("Starting training...")
    now = time.time()
    valid_loss = np.inf
    try:
        for epoch in train(iter_funcs, dataloader, x_, y_):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))

            if epoch['number'] >= num_epochs:
                break
            # we also save the best validation parameters:
            if epoch['valid_loss'] < valid_loss:
                valid_loss =  epoch['valid_loss']
                print "saving best validation parameters with best parameter epoch: "+str(epoch['number'])
                all_params = lasagne.layers.get_all_param_values(output_layer)
                saved_name = 'Lasagne_age_regressor_best.pkl'
                with(open(saved_name, 'wb')) as file:
                    pickle.dump(all_params, file)


    except KeyboardInterrupt:
        all_params = lasagne.layers.get_all_param_values(output_layer)
        with(open('Lasagne_age_regressor.pkl', 'wb')) as file:
            pickle.dump(all_params, file)

    all_params = lasagne.layers.get_all_param_values(output_layer)
    with(open('Lasagne_age_regressor.pkl', 'wb')) as file:
        pickle.dump(all_params, file)
    print "best parameter epoch: "+str(epoch['number'])
    return output_layer

if __name__ == '__main__':
    main()
'''
Epoch 195 of 200 took 2.453s
  training loss:		0.029527
  validation loss:		0.574878
Epoch 196 of 200 took 2.451s
  training loss:		0.030964
  validation loss:		0.577839
Epoch 197 of 200 took 2.451s
  training loss:		0.028449
  validation loss:		0.574156
Epoch 198 of 200 took 2.450s
  training loss:		0.028886
  validation loss:		0.569061
Epoch 199 of 200 took 2.451s
  training loss:		0.032668
  validation loss:		0.574900
Epoch 200 of 200 took 2.449s
  training loss:		0.030121
  validation loss:		0.588444
'''