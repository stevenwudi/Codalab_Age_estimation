
import numpy
import cPickle as pickle

predictions = pickle.load(open(r'D:\ChalearnAge\EUMSSI_data\predictions.pkl'))

predictions.keys()


y_prediction_n = predictions['y_prediction_n']
y_prediction_p = predictions['y_prediction_p']
X_falseAlarm_dir = predictions['X_falseAlarm_dir']
X_detectedPositive_dir = predictions['X_detectedPositive_dir']


n_index = numpy.argsort(y_prediction_n[:,1])
n_index = n_index[::-1]

p_index = numpy.argsort(y_prediction_p[:,1])



for i in range(10):
    print X_falseAlarm_dir[n_index[i]]
    print y_prediction_n[n_index[i], 1]


for i in range(10):
    print X_detectedPositive_dir[p_index[i]]
    print y_prediction_p[p_index[i], 1]


p_index = numpy.argsort(y_prediction_p[:,1])
p_index = p_index[::-1]

for i in range(20):
    print y_prediction_p[p_index[i], 1]