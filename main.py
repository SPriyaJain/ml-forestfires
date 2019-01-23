import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#cleaning
data_train = pd.read_csv('forestfires_cleaned.csv', header=0)
#print(len(data_train))

# data_train.replace('n', np.nan, inplace=True)
# data_train = data_train.dropna()
data_train = data_train[data_train['area'] != 0]

#print(len((data_train)))

#data prep
features = pd.get_dummies(data_train)
labels = np.array(data_train['area'])
data_train = data_train.drop('area', axis=1)
data_list = list(data_train.columns)
data_train = np.array(data_train)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)
#print('training features shape ', train_features.shape)
#print('labels shape', train_labels.shape)
#print('testing features shape', test_features.shape)
#print('testing labels shape', test_labels.shape)

#train
rf = RandomForestRegressor(n_estimators=800, random_state=42)
rf.fit(train_features,train_labels)

#test
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print(round(np.mean(errors), 2))

#confirm
mape = 100 * (errors/test_labels)
accuracy = 100 - np.mean(mape)
print(round(accuracy, 2))
