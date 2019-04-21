import numpy as np
np.random.seed(10)  #for reproducibility
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten,Dropout
from keras import optimizers

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
# X_train.shape (50000, 32, 32, 3)

# data pre-processing
#X_train = X_train.reshape(-1,3,32,32)
#X_test = X_test.reshape(-1,3,32,32)
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# build CNN
model=Sequential()

# Conv layer1 output shape (32,32,32)
model.add(Convolution2D(
    32,(3,3),
    strides=(1,1),
    input_shape=X_train.shape[1:],
    padding='same',
))
model.add(Activation('relu'))
model.add(Convolution2D(
    32,(3,3),
    strides=(1,1),
    padding='same',
    ))
model.add(Activation('relu'))

# Pooling layer1 (max pooling) output shape(32,16,16)
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides = (2,2)
    ))
model.add(Dropout(0.4))

# Conv layer2 output shape (32,32,32)
model.add(Convolution2D(
    32,(3,3),
    strides=(1,1),
    input_shape=X_train.shape[1:],
    padding='same',
))
model.add(Activation('relu'))
model.add(Convolution2D(
    32,(3,3),
    strides=(1,1),
    padding='same',
    ))
model.add(Activation('relu'))

# Pooling layer1 (max pooling) output shape(32,16,16)
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides = (2,2)
    ))
model.add(Dropout(0.2))

# Fully Connected layer1 input shape(64*8*8),output shape(1024)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

opt = keras.optimizers.rmsprop(lr=0.001,decay=0.0001)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_test /=255
#print(X_train[0])

print("Training---------------------\n")
model.fit(X_train,y_train,epochs=5,batch_size=64)

print("\nTesting-----------------------\n")

# Evaluate the model with the metrics we defined earlier
loss,accuracy = model.evaluate(X_test,y_test)

print("\nTest loss:",loss)
print("\nTest accuracy:",accuracy)



## define optimizer
#adam = Adam(lr=1e-4)

## Add metrics to get more result
#model.compile(optimizer=adam,
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

#print("Training¡­¡­")
## Train the model
#model.fit(X_train,y_train,nb_epoch=1,batch_size=32)

#print("\nTesting¡­¡­")
## Evaluate the model with the metrics we defined earlier
#loss,accuracy = model.evaluate(X_test,y_test)

#print("\nTest loss:",loss)
#print("\nTest accuracy:",accuracy)
