import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

s = "ZYX"
defect = ['Cracking', 'Ideal', 'Offset_Pulley', 'Wear']

def create_candles(df, bin_size=0.25):
    """
    Группирует данные по частоте с шагом bin_size и создаёт свечи:
    - Open: первое значение амплитуды в интервале
    - High: максимальное значение амплитуды
    - Low: минимальное значение амплитуды
    - Close: последнее значение амплитуды
    """
    # Округляем частоты до ближайшего интервала bin_size
    df = df.copy()
    df['bin'] = (df['Frequency_Hz'] // bin_size) * bin_size

    candle_data = []

    for bin_val, group in df.groupby('bin'):
        candle = {
            'Frequency_Hz': bin_val,
            'Open': group['Amplitude'].iloc[0],
            'High': group['Amplitude'].max(),
            'Low': group['Amplitude'].min(),
            'Close': group['Amplitude'].iloc[-1]
        }
        candle_data.append(candle)

    return pd.DataFrame(candle_data)

for j in defect:
    for i in range(3):
        folder_path = f"Fourier\\{j}\\{j}_{s[i]}\\"
        output_folder = f"Candles\\{j}\\{j}_{s[i]}\\"

        # Создаем папку для результатов, если она не существует
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                print(file_path)
                if os.path.isfile(file_path):
                    # 1. Загрузка данных из CSV
                    df = pd.read_csv(file_path)

                    # Проверка на наличие нужных столбцов
                    if 'Frequency_Hz' not in df.columns or 'Amplitude' not in df.columns:
                        print(f"Файл {file_path} не содержит нужных столбцов. Пропущен.")
                        continue

                    # 2. Создание свечей
                    candles_df = create_candles(df, bin_size=0.25)

                    # 3. Сохранение результата в CSV
                    output_file_path = os.path.join(output_folder, filename[:-4] + f"{s[i]}_candles.csv")
                    candles_df.to_csv(output_file_path, index=False)
                    print(f"Сохранено: {output_file_path}")

print("Обработка всех файлов завершена!")