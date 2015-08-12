import sys
pc = "linux"
#pc = "windows"
#pc = "virtualbox"
if pc=="linux": 
    sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
    net_params ='/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net_params.pickle'
    load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'
    fname = load_path+"data_with_label.hdf5"
elif pc=="windows":
    net_params = r'D:\ChalearnAge\net_params.pickle'
    fname ="data_with_label.hdf5"
    
    
from NN.Convnet_age_ony_regression import *
net_temp = ConvNet_Naive_Age_Regression()

#This is Di Wu's adaptation that because we have different last layer
#Though there is no difference between the "load_params_from" method
#because the constraint p1.get_value().shape == p2v.shape:
net_temp.net.load_params_from(net_params)


X, y, y_mean = load(fname)

net_temp.net.fit(X, y)


net_temp.net.save_params_to(load_path + 'net_params.pickle')




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

