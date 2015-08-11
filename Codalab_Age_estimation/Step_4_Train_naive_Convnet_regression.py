

from NN.Convnet_age_ony_regression import *
net_temp = ConvNet_Naive_Age_Regression()

#This is Di Wu's adaptation that because we have different last layer
#Though there is no difference between the "load_params_from" method
#because the constraint p1.get_value().shape == p2v.shape:
net_temp.net.load_params_from(r'D:\ChalearnAge\net_params.pickle')

fname ="data_with_label.hdf5"
X, y = load(fname)
net.fit(X, y)

X, _ = load2d(test=True)

net_temp.net.save_params_to(r'D:\ChalearnAge\net_params.pickle')

y_pred = net.predict(X)