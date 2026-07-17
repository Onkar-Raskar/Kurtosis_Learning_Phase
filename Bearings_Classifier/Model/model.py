import scipy.io
import numpy as np 
from scipy.stats import kurtosis, skew
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

normal_data_train = scipy.io.loadmat("Data/normal.mat")
faulty_data_train = scipy.io.loadmat("Data/faulty.mat")

normal_data_test = scipy.io.loadmat("Data/99.mat")
faulty_data_test = scipy.io.loadmat("Data/169.mat")

normal_train_signal = normal_data_train['X098_DE_time'].flatten()
faulty_train_signal = faulty_data_train['X106_DE_time'].flatten()

normal_test_signal = normal_data_test['X099_DE_time'].flatten()
faulty_test_signal = faulty_data_test['X169_DE_time'].flatten()

window_size = 2048
data_rows_train = []
data_rows_test = []

def extract_features(window, label):
    mean = np.mean(window)
    std = np.std(window)
    maximum = np.max(np.abs(window))

    rms = np.sqrt(np.mean(window**2))
    kurt = kurtosis(window)
    skewness = skew(window)
    p2p = np.ptp(window)

    crest_factor = maximum/rms
    mean_abs = np.mean(np.abs(window))
    shape_factor = rms/mean_abs
    impulse_factor = maximum/mean_abs
    clearance_factor = maximum/(np.mean(np.sqrt(np.abs(window)))**2)

    return {
        'Mean' : mean,
        'Std' : std,
        'RMS' : rms, 
        'Kurtosis': kurt, 
        'Peak_to_Peak' : p2p, 
        'Skewness' : skewness,
        'Crest_Factor' : crest_factor,
        'Shape_Factor' : shape_factor,
        'Impulse_Factor' : impulse_factor,
        'Clearance_Factor' : clearance_factor,
        'label' : label
    }

for i in range(0, len(normal_train_signal) - window_size, window_size):
    window  = normal_train_signal[i : i + window_size]
    data_rows_train.append(extract_features(window, label=0))

for i in range(0, len(faulty_train_signal) - window_size, window_size):
    window  = faulty_train_signal[i : i + window_size]
    data_rows_train.append(extract_features(window, label=1))

for i in range(0, len(normal_test_signal) - window_size, window_size):
    window  = normal_test_signal[i : i + window_size]
    data_rows_test.append(extract_features(window, label=0))

for i in range(0, len(faulty_test_signal) - window_size, window_size):
    window  = faulty_test_signal[i : i + window_size]
    data_rows_test.append(extract_features(window, label=1))

df_train = pd.DataFrame(data_rows_train)
df_test = pd.DataFrame(data_rows_test)

# print("DataSet Shape : ", df.shape)
# print(df.head())
# print("-"*30)
# print(df.tail())

features = ['Mean', 'Std', 'RMS', 'Kurtosis', 'Peak_to_Peak', 'Skewness' , 'Crest_Factor','Shape_Factor', 'Impulse_Factor', 'Clearance_Factor']

plt.scatter(df_train["RMS"],
            df_train["Kurtosis"],
            c=df_train["label"])
plt.xlabel("RMS")
plt.ylabel("Kurtosis")
plt.show()

x = df_train[['Mean', 'Std', 'RMS', 'Kurtosis', 'Peak_to_Peak', 'Skewness' , 'Crest_Factor','Shape_Factor', 'Impulse_Factor', 'Clearance_Factor']]
y = df['label']

x_train = df_train[features]
y_train = df_train[['label']]
x_test = df_test[features]
y_test = df_test[['label']]

print("Training the Random Forest")
model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split = 5, min_samples_leaf=2, class_weight="balanced" , bootstrap=True, random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("\n--confusion_matrix")
print(confusion_matrix(y_test,predictions))

print("\n--classification_report")
print(classification_report(y_test, predictions))