import os
import scipy.io
import numpy as np 
import matplotlib.pyplot as plt 

print(os.getcwd())

normal_data = scipy.io.loadmat("Data/normal.mat")
faulty_data = scipy.io.loadmat("Data/faulty.mat")

print("Keys in normal Data : " , normal_data.keys())
print("Keys in normal Data : " , faulty_data.keys())

normal_signal = normal_data['X098_DE_time'].flatten()
faulty_signal = faulty_data['X106_DE_time'].flatten()

print("Normal signal data")
print(normal_signal)
print("-"*30)
print("Faulty signal data")
print(faulty_signal)

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(normal_signal, color='blue')
plt.title("Normal bearing vibration")
plt.xlabel("Data Points (Time)")
plt.ylabel("Amplitude")
plt.ylim(-2,2)

plt.subplot(1,2,2)
plt.plot(faulty_signal, color='red')
plt.title("Faulty bearing vibration")
plt.xlabel("Data Points (Time)")
plt.ylabel("Amplitude")
plt.ylim(-2,2)

plt.tight_layout()
plt.show()
