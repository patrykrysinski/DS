from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:\dane\PATRYK\zero_to_deep_learning_video\data\weight-height.csv")

X = df['Height'].values
y = df['Weight'].values

model = Sequential()
model.add(Dense(1,input_shape = (1,))) ## Dense(yx) how many output y, how many input x
model.compile(Adam(learning_rate=0.8),'mean_squared_error')
model.fit(X,y,epochs=40)

y_pred = model.predict(X)
df.plot(x='Height',y='Weight',kind='scatter')
plt.plot(X,y_pred,color = 'red')

w,B = model.get_weights()

## train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y)

w[0,0] = 0 
B[0] = 0
model.set_weights((w,B))

model.get_weights()

model.fit(x_train,y_train,epochs=50)

## classification

df = pd.read_csv("C:\dane\PATRYK\zero_to_deep_learning_video\data\user_visit_duration.csv")