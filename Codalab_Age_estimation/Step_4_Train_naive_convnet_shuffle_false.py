import sys


sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
net_params ='/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net_params.pickle'
load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'
fname = load_path+"data_with_label_DPM.hdf5"


from NN.Convnet_age_ony_regression import *
net_temp = ConvNet_Naive_Age_Regression(shuffle=False)

#This is Di Wu's adaptation that because we have different last layer
#Though there is no difference between the "load_params_from" method
#because the constraint p1.get_value().shape == p2v.shape:

net_temp.net.load_params_from(net_params)


X, y, y_mean, y_std = load(fname)

net_temp.net.fit(X, y)


net_temp.net.save_params_to(load_path + 'net_params_DPM_NoShuffle.pickle')




'''
    990       0.00817       0.09543      0.08562  2.44s
    991       0.00775       0.09696      0.07991  2.44s
    992       0.00781       0.09747      0.08011  2.44s
    993       0.00780       0.09604      0.08119  2.44s
    994       0.00819       0.09557      0.08572  2.44s
    995       0.00785       0.09615      0.08165  2.46s
    996       0.00771       0.09759      0.07897  2.47s
    997       0.00807       0.09708      0.08311  2.46s
    998       0.00786       0.09586      0.08201  2.46s
    999       0.00803       0.09622      0.08340  2.47s
   1000       0.00825       0.09737      0.08478  2.46s
'''