
from NN.NeuralNetworks_kfkd import NeuralNetworks_kfkd


net = NeuralNetworks_kfkd()
net = net.net
net =  pickle.load(open('/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net.pickle','rb'))
