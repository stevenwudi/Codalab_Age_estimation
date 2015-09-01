import sys


sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
net_params ='/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net_params.pickle'
load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'
fname = load_path+"data_with_label_DPM.hdf5"


from NN.Convnet_age_ony_regression import *
net_temp = ConvNet_Naive_Age_Regression(shuffle=True)

#This is Di Wu's adaptation that because we have different last layer
#Though there is no difference between the "load_params_from" method
#because the constraint p1.get_value().shape == p2v.shape:

net_temp.net.load_params_from(net_params)


X, y, y_mean, std = load(fname)

net_temp.net.fit(X, y)


net_temp.net.save_params_to(load_path + 'net_params_DPM.pickle')




#   9991       0.00091       0.09985      0.00912  2.75s
#   9992       0.00092       0.10004      0.00917  2.75s
#   9993       0.00080       0.10028      0.00793  2.75s
#   9994       0.00088       0.10052      0.00876  2.75s
#   9995       0.00089       0.10067      0.00882  2.75s
#   9996       0.00096       0.10067      0.00958  2.75s
#   9997       0.00085       0.10059      0.00845  2.75s
#   9998       0.00084       0.10045      0.00834  2.75s
#   9999       0.00087       0.10025      0.00866  2.75s
#  10000       0.00088       0.10001      0.00880  2.75s

####################################
# DPM
####################################
'''
    990       0.00822       0.08671      0.09477  2.72s
    991       0.00817       0.08571      0.09531  2.72s
    992       0.00768       0.08575      0.08962  2.72s
    993       0.00787       0.08684      0.09063  2.72s
    994       0.00765       0.08733      0.08763  2.72s
    995       0.00780       0.08670      0.08999  2.72s
    996       0.00790       0.08638      0.09151  2.72s
    997       0.00769       0.08646      0.08898  2.72s
    998       0.00786       0.08686      0.09049  2.72s
    999       0.00790       0.08687      0.09089  2.72s
   1000       0.00806       0.08643      0.09326  2.72s
'''
