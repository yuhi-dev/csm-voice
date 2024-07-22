import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
plt.rcParams['figure.dpi'] = 100 

# 音声ファイルの読み込み
snd = parselmouth.Sound("waves/result/test_wave.wav")

# スペクトログラムを描画する関数
def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

# 音の強度を描画する関数
def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

# ピッチを描画する関数
def draw_pitch(pitch):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

# フォルマントを描画する関数
def draw_formants(formants):
    for num in range(1, 5):
        plt.plot(formants.index, formants[num], marker='o', linestyle='-', markersize=4, label=f"F{num}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Formants")
    plt.legend()

# スペクトログラムを作成
intensity = snd.to_intensity()
spectrogram = snd.to_spectrogram()
pitch = snd.to_pitch()  # ピッチを取得

# スペクトログラムとフォルマントを重ねて描画
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_intensity(intensity)
draw_pitch(pitch)  # ピッチを重ねて描画
plt.xlim([snd.xmin, snd.xmax])
plt.show()
