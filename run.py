from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.preprocessing import image

################################################################################################################
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.load_weights('covid.hdf5')

################################################################################################################
img_pred = image.load_img("test pic/covid.jpeg", target_size= (200, 200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

rslt = model.predict(img_pred)
if rslt[0][0] == 1:
    prediction = "Normal"
else:
    prediction = "Covid - 19"
print(prediction)

img_pred = image.load_img("test pic/normal.jpeg", target_size= (200, 200))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

rslt1 = model.predict(img_pred)
if rslt1[0][0] == 1:
    prediction = "Normal"
else:
    prediction = "Covid - 19"
print(prediction)
