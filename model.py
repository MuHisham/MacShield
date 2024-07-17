import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('PM_Dataset.csv')

inputCol = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
outputCol = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
labels = ['Tool Wear Failure', 'Heat Dissipation Failure', 'Power Failure', 'Overstrain Failure', 'Random Failure']

data['FType'] = 0
for i in range(len(data)):
    if data.iloc[i]['TWF'] == 1:
        data.at[i, 'FType'] = 1
    elif data.iloc[i]['HDF'] == 1:
        data.at[i, 'FType'] = 2
    elif data.iloc[i]['PWF'] == 1:
        data.at[i, 'FType'] = 3
    elif data.iloc[i]['OSF'] == 1:
        data.at[i, 'FType'] = 4
    elif data.iloc[i]['RNF'] == 1:
        data.at[i, 'FType'] = 5

data.to_csv('modifiedData.csv', index=False)

x = data[inputCol].values
y = data['FType'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

joblib.dump(svm_model, 'svm_model.pkl')

# loaded_svm_model = joblib.load('../PM_WebHost/svm_model.pkl')


# Getting input from Flutter
# new_input = [[298.3, 308.1, 1412, 52.3, 218]]
# # Getting output from Python
# predicted_category = int(loaded_svm_model.predict(new_input))
# print(labels[predicted_category - 1])


