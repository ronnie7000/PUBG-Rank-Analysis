import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train_V2.csv')
test = pd.read_csv('test_V2.csv')

rnd_data_test = test.sample(frac = 0.005)

rnd_data = train.sample(frac = 0.005 )

rnd_data = rnd_data.iloc[:,3:]

rnd_data_test = rnd_data_test.iloc[:,3:]


rnd_data = rnd_data.drop(labels = ["matchType","swimDistance","vehicleDestroys"], axis = 1 )
rnd_data_test = rnd_data_test.drop(labels = ["matchType","swimDistance","vehicleDestroys"], axis = 1 )

X_test = rnd_data_test.iloc[ : , : 21]
y_test = rnd_data_test.iloc[ : , -1]


X = rnd_data.iloc[ : , : 21]
Y = rnd_data.iloc[ : , -1]


from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(21, input_dim=21, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(X, Y, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)


print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print(score)
