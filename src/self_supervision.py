# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 22:21:32 2022
version2
@author: haoyu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:12:59 2022
self-supervised/contrastive learning for the F

build a multi-task model to learn different data-augmentation tasks
pre-train with wisdm dataset
@author: haoyu
"""
import numpy as np
from sklearn import preprocessing
from keras.layers import Input, Conv2D,  Dense, Flatten,MaxPooling2D, concatenate
from keras.models import Model, Sequential
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from keras.layers import LSTM, Permute,Reshape
import keras
from sklearn.model_selection import train_test_split
import collections
import tensorflow as tf

np.random.seed(101)

#num of training epochs
nEpochs = 100

#output of each task-specific layer is either 0 or 1
NUM_CLASSES = 1

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 200

# Batch Size
BATCH_SIZE = 32

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

NUM_CHANNEL = 6

# =============================================================================
# Construct the neural network
# =============================================================================
#The 'Net' part, will be replaced by 4CNN+2LSTM

inputs_1 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_1')
inputs_2 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_2')
inputs_3 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_3')
inputs_4 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_4')
inputs_5 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_5')
inputs_6 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_6')
inputs_7 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_7')
inputs_8 = Input((1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL, ), name='input_8')

def create_base_network():
    """
    Base network to be shared.
    """    
    multi_input = Input(shape=(1, SLIDING_WINDOW_LENGTH,NUM_CHANNEL), name='multi_input')
    print(multi_input.shape)  # (?, 1, 24, 113)
    
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(multi_input)
    print(y.shape)  # (?, 64, 20, 113)
    
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    y = Conv2D(64, (5, 1), activation='relu', data_format='channels_first')(y)
    print(y.shape)
    
    y = Permute((2, 1, 3))(y)
    print(y.shape)  # (?, 20, 64, 113)
    
    # This line is what you missed
    # ==================================================================
    y = Reshape((int(y.shape[1]), int(y.shape[2]) * int(y.shape[3])))(y)

    # ==================================================================
    print(y.shape)  # (?, 20, 7232)
    
    y = LSTM(128,dropout=0.1,return_sequences=True)(y)
    y = LSTM(128,dropout=0.1)(y)
      # (?, 128)
    
    return keras.Model(inputs=multi_input, outputs=y)  



# Shared embedding layer for positive and negative items
Shared_DNN = create_base_network()
feature_1 = Shared_DNN(inputs_1)
feature_2 = Shared_DNN(inputs_2)
feature_3 = Shared_DNN(inputs_3)
feature_4 = Shared_DNN(inputs_4)
feature_5 = Shared_DNN(inputs_5)
feature_6 = Shared_DNN(inputs_6)
feature_7 = Shared_DNN(inputs_7)
feature_8 = Shared_DNN(inputs_8)
# merged_vector = concatenate([feature_1, feature_2, feature_3,feature_4,
#                              feature_5,feature_6,feature_7,feature_8], axis=-1, name='merged_layer')


#Define 8 task specific layer for binary classification
finalAct = 'sigmoid'
#HIDDEN_NUM = 256
sub1 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_1)
sub2 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_2)
sub3 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_3)
sub4 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_4) 

sub5 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_5)
sub6 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_6)
sub7 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_7)
sub8 = Dense(units=NUM_CLASSES,use_bias=True,activation=finalAct)(feature_8) 


model = Model(inputs=[inputs_1,inputs_2,inputs_3,inputs_4,inputs_5,inputs_6,inputs_7,inputs_8], 
              outputs=[sub1,sub2,sub3,sub4,sub5,sub6,sub7,sub8])

#define loss as binary cross entropy plus L2 norm for each layer
def multi_task_loss(model, mu = 0.0001):
    def _loss(y_true, y_pred):
        myloss = K.mean(K.binary_crossentropy(y_true, y_pred))
        vars_   = tf.trainable_variables() 
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_ ])
        return myloss + lossL2 * mu
    return _loss

tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)
adam_optim = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999)
#model.compile(loss=multi_task_loss(model), 
model.compile(loss='binary_crossentropy',
              metrics=[tf.keras.metrics.BinaryAccuracy()],
              optimizer=adam_optim)
print(model.summary())

# =============================================================================
# load data
# =============================================
def load_tensor(name):
    # Read the array from disk
    new_data = np.loadtxt('../data/7act_wisdm/Sub'+str(name)+'_data.txt')
    
    # Note that this returned a 2D array!
    print (new_data.shape)
    
    # However, going back to 3D is easy if we know the 
    # original shape of the array
    new_data = new_data.reshape((-1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    return new_data

def load_data_by_id(subid):
    X = load_tensor(subid)
    y = np.loadtxt("../data/7act_wisdm/Sub"+str(subid)+"_label.txt",dtype=int)
    X_train,X_test,y_train,y_test = train_test_split(X, y, 
                                                    train_size=0.8, 
                                                    random_state=42,
                                                    stratify=y)
    return X_train,X_test,y_train,y_test


labels = np.array([0,1,2,3,4,5,6])

#1.load and divide data for each subject in the 16 subjects
X_train,X_val,y_train,y_val = [],[],[],[]
for i in range(1600,1636):
    xtrain,xval,ytrain,yval = load_data_by_id(i)
    X_train.append(xtrain)
    y_train.append(ytrain)
    X_val.append(xval)
    y_val.append(yval)
print(len(X_train))
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

# =============================================================================
# discard original label and make data transformations
# 1: transformed 0: original
# =============================================================================
from imu_transformation import DA_Jitter, DA_Scaling, DA_Rotation, DA_Negated, DA_Horizontal_flip, DA_Permutation, DA_TimeWarp, DA_channel_shuffle
from tools import plot_raw
# # Shuffle before fetch batch.
# features, labels = (X_val, y_val)
# dataset = tf.data.Dataset.from_tensor_slices((features,labels))
# dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
# #fetch data from dataset
# dataset_as_list = list(dataset.as_numpy_iterator())

# #generate inputs to NN
# #random idx within a batch, to make half as transformed, half as not
# indices = np.random.permutation(64)[:32]


def make_xtrain(X_train):
    num_sample = X_train.shape[0]
    #random select samples for transform
    indices = np.random.permutation(num_sample)[:int(num_sample//2)]
    # raw = np.delete(X_train, indices, axis=0) #useful for generating all 8 inputs
    # inputs_transform = DA_Jitter(X_train[indices])
    # inputs_1 = np.vstack((inputs_transform,raw))
    inputs_1 = DA_Jitter(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_2 = DA_Scaling(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_3 = DA_Rotation(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_4 = DA_Negated(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_5 = DA_Horizontal_flip(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_6 = DA_Permutation(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_7 = DA_TimeWarp(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    inputs_8 = DA_channel_shuffle(X_train, indices).reshape((-1,1,SLIDING_WINDOW_LENGTH,NUM_CHANNEL))
    
    #make target onehot vec
    outputs = np.zeros((num_sample,))
    outputs[indices] = 1
    lb = preprocessing.LabelBinarizer()
    lb.fit(y=outputs)
    
    y_onehot = lb.transform(outputs)
    target_onehot = []
    for i in range(8):
        target_onehot.append(y_onehot.astype(np.float32))   #require float32 by tf
        
    return inputs_1, inputs_2, inputs_3, inputs_4, inputs_5, inputs_6, inputs_7, inputs_8, target_onehot
    
inputs_1, inputs_2, inputs_3, inputs_4, inputs_5, inputs_6, inputs_7, inputs_8, target_onehot = make_xtrain(X_train)
val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_onehot = make_xtrain(X_val)
# =============================================================================
# train the network and save
# =============================================================================

with tf.device('/gpu:0'): 
    model.fit([inputs_1,inputs_2,inputs_3,inputs_4,inputs_5,inputs_6,inputs_7,inputs_8],
          y=target_onehot,
          validation_data=([val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8],
                                   val_onehot), 
          batch_size=BATCH_SIZE, 
          verbose=1,
            shuffle = True,
            epochs=nEpochs)
    
# serialize model to JSON
model_json = model.to_json()
with open("pretrain_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("pretrain_model_weight.h5")
print("Saved model to disk")

