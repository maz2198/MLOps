"""
Source: https://www.kaggle.com/shivam2503/diamonds

Description
===========
carat: weight of the diamond (0.2--5.01)
cut: quality of the cut (Fair, Good, Very Good, Premium, Ideal)
color: diamond colour, from J (worst) to D (best)
clarity: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
table: width of top of diamond relative to widest point (43--95)
price: price in US dollars (\$326--\$18,823)
x: length in mm (0--10.74)
y: width in mm (0--58.9)
z: depth in mm (0--31.8)
SIZE
====
(53940, 11)
id   carat   cut   color   clarity   depth   table   price   x      y      z
0    0.23    6.0   6       2         61.5    55.0    326     3.95   3.98   2.43
1    0.21    5.0   6       3         59.8    61.0    326     3.89   3.84   2.31
2    0.23    4.0   6       5         56.9    65.0    327     4.05   4.07   2.31
3    0.29    5.0   2       4         62.4    58.0    334     4.20   4.23   2.63
4    0.31    4.0   1       2         63.3    58.0    335     4.34   4.35   2.75
...
"""


import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, VotingClassifier


diamonds = pd.read_csv('diamonds.csv')
diamonds.drop(['id'], axis=1, inplace=True)

# Handling the ordinal variables.
cut_dict = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_dict = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_dict = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

diamonds['cut'] = diamonds['cut'].map(cut_dict)
diamonds['color'] = diamonds['color'].map(color_dict)
diamonds['clarity'] = diamonds['clarity'].map(clarity_dict)

# Getting X and Y data.
X_data = diamonds.drop('price', axis=1).values
y_data = diamonds[['price']].values

# Defining the scalers to scale the X and the Y data.
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Scaling the X and the Y data.
X_scaled_data = X_scaler.fit_transform(X_data)
Y_scaled_data = Y_scaler.fit_transform(y_data)

# Splitting the data on training and testing parts (70%/30%).
X_train, X_test, y_train, y_test = train_test_split(X_scaled_data, Y_scaled_data,
                                                    test_size=0.3, random_state=42)

# Defining the model's parameters.
learning_rate = 0.001
training_epochs = 300

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Defing the model.

MLP = MLPRegressor(hidden_layer_sizes=(layer_1_nodes, layer_2_nodes, layer_3_nodes),
                     activation='relu',
                     solver='adam',
                     learning_rate_init=learning_rate,
                     max_iter=training_epochs,
                     shuffle=True,)


rf = RandomForestRegressor(random_state=42,n_jobs=-1,n_estimators=layer_2_nodes,max_depth=12,min_samples_split=2)

# Creating Ensemble Model Soft VotingClassifier
model = VotingClassifier([('MLP',MLP),('rf',rf)],voting='soft',weights=[2,1])
# Training of the model.
model.fit(X_train, np.ravel(y_train))

print('Training is complete.')
print('Final Training Score (R^2): {:.3f}'.format(model.score(X_train, y_train)))
print('Final Test Score (R^2): {:.3f}'.format(model.score(X_test, y_test)))

y_predicted_scaled = model.predict(X_test)
y_predicted = Y_scaler.inverse_transform(y_predicted_scaled.reshape(-1,1))
y_real = Y_scaler.inverse_transform(y_test)

for i in range(20):
    print('Real value: {} predicted value {}'.format(y_real[i], y_predicted[i]))
