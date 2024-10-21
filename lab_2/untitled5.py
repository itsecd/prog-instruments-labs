
from scipy import signal
from scipy.io import wavfile
import sklearn.neural_network as nn
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.metrics as mt
import matplotlib.pyplot as plt
import os
import numpy as np

def write_to_csv(file_names,output_file,data_array):
    output = open(output_file, 'w')
    output.write(",")
    for i in file:
        output.write(i+',')
    for i in range(67):
        output.write('\n'+file_names[i])
    for j in range(67):
        output.write(','+str(data_array[i,j]))
    output.close()

def prepare_and_print(data_array,labels_array,metric_type):
    data_array  = (data_array-np.min(data_array))/(np.max(data_array)-np.min(data_array))#normalize data
    data_array_t = np.transpose(data_array)                                         #transpose to get a second array in a pair KL(x,y)
    data_array =np.triu(data_array)                                                 #remove duplicate pairs
    data_array_t =np.triu(data_array_t)                                   
    data_array_label1 = conditions_label* data_array                       
    data_array_label2 = data_array-data_array_label1
    data_array_t_label1 = conditions_label* data_array_t
    data_array_t_label2 = data_array_t-data_array_t_label1                          #get arrays of pairs for each label
    
    plt.figure()
    plt.xlabel(metric_type+'(a,b)')
    plt.ylabel(metric_type+'(b,a)')
    plt.scatter(data_array_label1, data_array_t_label1,s=0.5)
    plt.scatter(data_array_label2, data_array_t_label2,s=0.5)
    plt.show()
    
def make_class_labels(file_names):
    class_labels = []
    for i in file_names:
        label_tmp=0
        if 'iphone' in i:
            label_tmp+=2
        if 'kitchen' in i:
            label_tmp+=1  #iphone-kitchen=3 iphone-room=2 android kitchen = 1 android room = 0
        class_labels.append(label_tmp)
    return np.array(class_labels) 

def separate_by_conditions(class_labels):
    conditions_label = []
    for i in range(class_labels.shape[0]):
        for j in range(i,class_labels.shape[0]):
            if class_labels[i]==class_labels[j]:
                conditions_label.append(1)
            else:
                conditions_label.append(0)
    return conditions_label

def get_metrics(y_true, y_pred):
    acc = mt.accuracy_score(y_true, y_pred)
    precision = mt.precision_score(y_true, y_pred)
    rcall = mt.recall_score(y_true, y_pred)
    f1 = mt.f1_score(y_true, y_pred)
    return acc,f1,rcall,precision

wav_files=[]
file =[]
written=0
for r, d, f in os.walk('dataset1/'):
    for i in f:
        wav_files.append(i)
        file.append(i)
    for i in file:
        samprate,data = wavfile.read(os.path.join(r,i))
        if (len(np.shape(data))==2):
            data_size = np.shape(data)[0]
            data = np.ravel(data)
            data=data[:data_size]
        written=written+1
        wavfile.write(os.path.join(r,i),samprate,data)

            
wav_files=[]
freq_and_pow = {}
for r, d, f in os.walk('my_set/'):
    for i in f:
        wav_files.append(i)
for i in wav_files:
    samprate,data = wavfile.read(os.path.join(r,i))
    freq,Pow = signal.welch(data,samprate,nperseg=2048)
  #  plt.semilogy(freq,Pow)
   # plt.xlabel('Частота в Hz')
   # plt.ylabel('Спектральная мощность')
   # plt.show()
    freq_and_pow[i]={}
    freq_and_pow[i]['Sample rate']=samprate
    freq_and_pow[i]['frequency_size']=np.size(freq)
    freq_and_pow[i]['Power_size']=np.size(Pow)
    freq_and_pow[i]['Data_size']=np.shape(data)
    
output = open('dataset2.csv', 'w')
output.write("Название файла,")
output.write('Частота дискретизации,')
output.write('Формат по частоте,')
output.write('Формат по мощности')
output.write('Формат данных')
for key in freq_and_pow.keys():
    output.write('\n'+key+',')
    for column in freq_and_pow[key].keys():
        output.write(str(freq_and_pow[key][column])+',')
output.close()

wav_files=[]
power = []
freqs = []
for r, d, f in os.walk('my_set/'):
    for i in f:
        wav_files.append(os.path.join(i))
    for i in wav_files:
        samprate,data = wavfile.read(os.path.join(r,i))
        freq,Pow = signal.welch(data,samprate)
        Pow = Pow/np.sum(Pow)
        power.append(Pow)
        freqs.append(freq)
        
power=np.array(power)

KL_np = np.zeros([67,67])
KL_val = 0
for i in range(67):
    for j in range(67):
        KL_val=np.sum(power[i]*np.log2(power[i]/power[j]))
        KL_np[i,j]=KL_val      
        KL_val = 0


IS_np = np.zeros([67,67])
IS_val = 0
for i in range(67):
    for j in range(67):
        IS_val=np.sum(power[i]/power[j]+np.log2(power[i]/power[j])-1) #правильно?
        IS_np[i,j] = IS_val
        IS_val =0
    
class_label = make_class_labels(file)    
conditions_label = separate_by_conditions(class_label)
    
write_to_csv(file, 'results_KL.csv', KL_np)
write_to_csv(file, 'results_IS.csv', IS_np)

#prepare_and_print(KL_np,conditions_label,'KL')
#prepare_and_print(IS_np,conditions_label,'IS')

KL_np_ab=[]
KL_np_ba=[]
IS_np_ab=[]
IS_np_ba=[]

for i in range(67):
    for j in range(i,67):
        KL_np_ab.append(KL_np[i,j])
        KL_np_ba.append(KL_np[j,i])
        IS_np_ab.append(IS_np[i,j])
        IS_np_ba.append(IS_np[j,i])

KL_np_ab=np.array(KL_np_ab)
KL_np_ba=np.array(KL_np_ba)
IS_np_ab=np.array(IS_np_ab)
IS_np_ba=np.array(IS_np_ba)


KL_np_ab_train,KL_np_ab_test,KL_np_ba_train,KL_np_ba_test,IS_np_ab_train,IS_np_ab_test,IS_np_ba_train,IS_np_ba_test,label_train,label_test = ms.train_test_split(np.ravel(KL_np_ab),
                                                                                                                                                                np.ravel(KL_np_ba),
                                                                                                                                                                np.ravel(IS_np_ab),
                                                                                                                                                                np.ravel(IS_np_ba),
                                                                                                                                                                np.ravel(conditions_label))
features_train = np.vstack((KL_np_ab_train,KL_np_ba_train,IS_np_ab_train,IS_np_ba_train))
features_train = np.transpose(features_train)
features_test = np.vstack((KL_np_ab_test,KL_np_ba_test,IS_np_ab_test,IS_np_ba_test))
features_test = np.transpose(features_test)
#label_train = label_train.reshape(-1,1)
#label_test = label_test.reshape(-1,1)

pp.normalize(features_train)
pp.normalize(features_test)

MLP =  nn.MLPClassifier().fit(features_train,label_train)
print(MLP.get_params())
print(MLP.score(features_test,label_test))
metrics = get_metrics(label_test, MLP.predict(features_test))
predict = MLP.predict(features_test)
roc_x,roc_y, threshold = mt.roc_curve(label_test, MLP.predict_proba(features_test)[:,1],pos_label=1)
plt.figure()
plt.title('ROC-кривая')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(roc_x,roc_y)
plt.plot([0,1],[0,1])
plt.show()
class_report = mt.classification_report(label_test, MLP.predict(features_test))
print(class_report)