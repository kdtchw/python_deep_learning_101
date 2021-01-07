import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=X.shape[1])
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)



from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o", "s"]
colors = ["r", "g"]
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
    
mappings = som.win_map(X)
# mapping coordinates of the white grid in som
frauds = np.concatenate((mappings[(4,5)], mappings[(6,7)]), axis=0)
frauds = sc.inverse_transform(frauds)


customers = dataset.iloc[:, 1:].values

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1
        
        
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=customers.shape[1]))

classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

classifier.fit(customers, is_fraud, batch_size=1, epochs=10)

y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis=1)

y_pred = y_pred[y_pred[:,1].argsort()]