
# we need to import all the classes
from NN.NeuralNetworks_kfkd import *


# pre-defined NN class 
net_temp = NeuralNetworks_kfkd()


load_path= '/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net.pickle'
net_temp.load_params(load_path)

