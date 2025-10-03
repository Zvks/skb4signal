import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
s = "ZYX"
defect = ['Cracking', 'Ideal', 'Offset_Pulley', 'Wear']
for j in defect:
    for i in range(3):
        folder_path = f"Data_prepared\\{j}\\{j}_{s[i]}\\"
        output_folder = f"Data_prepared\\Fourier\\{j}\\{j}_{s[i]}\\"  # папка для сохранения результатов

        # Создаем папку для результатов, если она не существует
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                print(file_path)
                if os.path.isfile(file_path):
                    # 1. Загрузка данных из CSV (исправлено - убраны кавычки вокруг file_path)
                    df = pd.read_csv(file_path)
                    signal = df[f'Channel {i+1}'].values

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
                    plt.title(f'Оригинальный сигнал - {filename}')
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

                    plt.tight_layout()
                    
                    # Сохраняем график
                    plot_filename = f"fourier_{Path(filename).stem}.png"
                    plot_path = os.path.join(output_folder, plot_filename)
                    #plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()  # закрываем figure чтобы не накапливать в памяти
                    
                    # Сохраняем данные FFT в CSV
                    fft_df = pd.DataFrame({
                        'Frequency_Hz': xf,
                        'Amplitude': amplitude
                    })
                    fft_filename = f"{j}_fft_{Path(filename).stem}_{s[i]}.csv"
                    fft_path = os.path.join(output_folder, fft_filename)
                    fft_df.to_csv(fft_path, index=False)
                    
                    print(f"Обработан файл: {filename}")

print("Обработка всех файлов завершена!")