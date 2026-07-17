import scipy.io
import numpy as np 
from scipy.stats import kurtosis, skew
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

normal_data = scipy.io.loadmat("Data/normal.mat")
faulty_data = scipy.io.loadmat("Data/faulty.mat")

normal_signal = normal_data['X098_DE_time'].flatten()
faulty_signal = faulty_data['X106_DE_time'].flatten()

window_size = 2048
data_rows = []

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

for i in range(0, len(normal_signal) - window_size, window_size):
    window  = normal_signal[i : i + window_size]
    data_rows.append(extract_features(window, label=0))

for i in range(0, len(faulty_signal) - window_size, window_size):
    window  = faulty_signal[i : i + window_size]
    data_rows.append(extract_features(window, label=1))

df = pd.DataFrame(data_rows)

print("DataSet Shape : ", df.shape)
print(df.head())
print("-"*30)
print(df.tail())

x = df[['Mean', 'Std', 'RMS', 'Kurtosis', 'Peak_to_Peak', 'Skewness' , 'Crest_Factor','Shape_Factor', 'Impulse_Factor', 'Clearance_Factor']]
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print("Training the Random Forest")
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("\n--confusion_matrix")
print(confusion_matrix(y_test,predictions))

print("\n--classification_report")
print(classification_report(y_test, predictions))