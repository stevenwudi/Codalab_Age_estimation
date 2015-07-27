
from NN import NeualNetworks_fkfd


#with open('net.pickle', 'wb') as f:
 #   pickle.dump(net, f, -1)# -*- coding: utf-8 -*-

net =  pickle.load(open('/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net.pickle','rb'))
