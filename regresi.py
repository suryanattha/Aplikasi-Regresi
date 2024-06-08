import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data
data = pd.read_csv('Student_Performance.csv')

# Mengambil kolom yang relevan
X = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Model Linear
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Model Eksponensial
X_log = np.log(X + 1)  # Menggunakan logaritma dari X
exp_model = LinearRegression()
exp_model.fit(X_log, y)
y_pred_exp = exp_model.predict(X_log)

# Plot grafik titik data dan hasil regresi
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data asli')
plt.plot(X, y_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Data asli')
plt.plot(X, y_pred_exp, color='green', label='Regresi Eksponensial')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Eksponensial')
plt.legend()

plt.show()

# Menghitung galat RMS
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
rms_exp = np.sqrt(mean_squared_error(y, y_pred_exp))

print(f'Galat RMS Regresi Linear: {rms_linear}')
print(f'Galat RMS Regresi Eksponensial: {rms_exp}')
