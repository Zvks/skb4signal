import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "Data_prepared\\Cracking\\Cracking_X\\M(1).csv"
df = pd.read_csv(file_path)

df_34 = df.iloc[(len(df)//4):(len(df)//4*3)]
signal = df_34["Channel 3"]

# 2. Параметры сигнала
N = len(signal)           # количество точек
T = 1.0 / 679.0          # шаг дискретизации
Fs = 1 / T  # частота дискретизации

# 3. Построение оригинального графика (временная область)
time = np.linspace(0.0, N*T, N, endpoint=False)

plt.figure(figsize=(10, 8))

# Оригинальный сигнал (вверху)
plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title(f'Оригинальный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid()

# 4. Вычисление FFT
yf = np.fft.fft(signal)
xf = np.fft.fftfreq(N, T)[:N//2]

# 5. Амплитудный спектр
amplitude = 2.0 / N * np.abs(yf[:N//2])

# 6. Построение амплитудного спектра
plt.subplot(2, 1, 2)
plt.plot(xf, amplitude)
plt.title('Амплитудный спектр (FFT)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()
# Сохраняем данные FFT в CSV
fft_df = pd.DataFrame({
    'Frequency_Hz': xf,
    'Amplitude': amplitude
    })
fft_filename = "Cracking_X_M(1)_centre.csv"
fft_df.to_csv(fft_filename, index=False)

plt.tight_layout()