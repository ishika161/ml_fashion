'''creating a identification model which identifies fashion accessories'''
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random



'''importing grey scale images'''
#import dataset
fashion_train_df=pd.read_csv('fashion-mnist_train.csv',sep=',')
fashion_test_df=pd.read_csv('fashion-mnist_test.csv',sep=',')

#visualise dataset
'''visualise top 5 rows'''
fashion_train_df.head()
'''visualise last 5 rows'''
fashion_train_df.tail()
fashion_test_df.head()
fashion_test_df.tail()
'''gives number of dimension in array'''
fashion_train_df.shape
fashion_test_df.shape
'''converting df to array'''
training=np.array(fashion_train_df,dtype='float32')
testing=np.array(fashion_test_df,dtype='float32')
i=random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))
label=training[i,0]
label
"The 10 classes are as follows:  \n",
"0 => T-shirt/top\n",
"1 => Trouser\n",
"2 => Pullover\n",
"3 => Dress\n",
"4 => Coat\n",
"5 => Sandal\n",
"6 => Shirt\n",
"7 => Sneaker\n",
"8 => Bag\n",
"9 => Ankle boot\n",

w_grid=15
l_grid=15
fig,axes=plt.subplots(l_grid,w_grid,figsize=(17,17))
axes=axes.ravel()
n_training=len(training)
for i in np.arange(0,w_grid*l_grid):
    index=np.random.randint(0,n_training)
    axes[i].imshow(training[index,1:].reshape(28,28))
    axes[i].set_title(training[index,0],fontsize=8)
    axes[i].axis('off')   
plt.subplots_adjust(hspace=0.4)

#training the model(cnn)
X_train=training[:,1:]/255
y_train=training[:,0]
X_test=testing[:,1:]/255
y_test=testing[:,0]
#validation dataset(to avoid overfitting)
from sklearn.model_selection import train_test_split
X_train,X_validate,y_train,y_validate=train_test_split(X_train,y_train,test_size=0.2,random_state=12345)
#reshape data
X_train=X_train.reshape(X_train.shape[0],*(28,28,1))
X_test=X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate=X_validate.reshape(X_validate.shape[0],*(28,28,1))
X_train.shape
X_test.shape
X_validate.shape

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

#building model
cnn_model=Sequential()
#using 32 faeture dectectors each of dimendion 3,3
cnn_model.add(Conv2D(32,3,3,input_shape=(28,28,1),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32,activation='relu'))
cnn_model.add(Dense(output_dim=10,activation='sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
cnn_model.fit(X_train,y_train,batch_size=512,epochs=25,verbose=1,
              validation_data=(X_validate,y_validate))

cnn_model.summary()
#eavaluate model 
evaluation=cnn_model.evaluate(X_test,y_test)
print('test accuracy: {:.3f}'.format(evaluation[1]))
predict_classes=cnn_model.predict_classes(X_test)

W=5
L=5
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()

for i in np.arange(0,L*W):
   
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("prediction class={:0.1f}\n true class={:0.1f}".format(predict_classes[i],y_test[i]))
    axes[i].axis('off')   
plt.subplots_adjust(hspace=0.5)
