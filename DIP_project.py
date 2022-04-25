from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
plt.imshow(X_train[0])
plt.title(y_train[0])
plt.show()

#if we want to see image in gray scale
plt.imshow(X_train[300],cmap='gray')
plt.title(y_train[300])
plt.show()

#create model
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=4, kernel_size=(7,7),activation='relu',input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=10,activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',metrics=['acc'])

model.summary()

print(model.fit(X_train, y_train, epochs=1, batch_size=1))

plt.imshow(X_test[0])
test = X_test[0].reshape(-1,28,28,1)
prediction =model.predict(test)
classes_x = np.argmax(prediction,axis=1)
print(classes_x)
plt.show()

model.save('mytrainedmodel.h5')
