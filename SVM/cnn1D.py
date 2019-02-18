from __future__ import print_function
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from utilities.data_reader import load_data2

fname = "../data/20171101_RAW_export.xyz"
# fname = "../data/stendalmark_20181120_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 23)

from sklearn.model_selection import train_test_split
X_train, X_test, lbl_train, lbl_test = train_test_split(dbdt, lbl, test_size = 0.30)

X_train = numpy.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
X_test = numpy.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
print(X_train.shape)

n_timesteps = X_train.shape[0]
n_features = X_train.shape[1]
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(None,n_features, X_train.shape[2])))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, lbl_train, epochs=10, batch_size = 64, verbose = 1)

scores = model.evaluate(X_test, lbl_test, verbose=0)
lbl_pred_scor = model.predict(X_test)
lbl_pred = lbl_pred_scor.round()
print(lbl_pred)
print("Accuracy: %.2f%%" % (scores[1] * 100))
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(lbl_test, lbl_pred))
print(classification_report(lbl_test, lbl_pred))