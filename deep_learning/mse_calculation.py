import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\dane\PATRYK\zero_to_deep_learning_video\data\weight-height.csv")

X = df['Height'].values
y = df['Weight'].values

def line(x: list,w: float=0,b: float=0) -> list: 
    return x*w +b

def mse(y_true: np.array, y_pred: np.array) ->float:
    return ((y_true - y_pred)**2).mean()

def lr_optimize(X,y,bs):
    mse_list = []
    for b in bs:
        y_hat = line(X,w=2,b=b)
        mse_list.append(mse(y,y_hat))
        plt.plot(X,y_hat)
    return mse_list

ax1 = plt.subplot(121)
bs=[-100,-50,0,50,100]
df.plot(kind='scatter',x='Height',y='Weight',ax=ax1)
mse_list = lr_optimize(X,y,bs)
ax2 = plt.subplot(122)
plt.plot(bs, mse_list)