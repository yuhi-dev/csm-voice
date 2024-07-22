import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set()
plt.rcParams['figure.dpi'] = 100 

snd = parselmouth.Sound("waves/result/saibameru_vocalo.wav")

plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show()

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

intensity = snd.to_intensity()
spectrogram = snd.to_spectrogram()
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_intensity(intensity)
plt.xlim([snd.xmin, snd.xmax])
plt.show()

def draw_pitch(pitch):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

pitch = snd.to_pitch()

pre_emphasized_snd = snd.copy()
pre_emphasized_snd.pre_emphasize()
spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)

plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show()

def draw_formants(formants):
    for num in range(1, 5):
        plt.plot(formants.index, formants[num], marker='o', linestyle='-', markersize=4, label=f"F{num}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Formants")
    plt.legend()
    plt.show()

# フォルマント分析
formants_burg = snd.to_formant_burg(max_number_of_formants=5.0, maximum_formant=12000.0, window_length=0.025, pre_emphasis_from=50.0)

# 各時刻における第1~第4フォルマントを取得する
# pandasを使って成形する
formants = {i: [] for i in range(1, 5)}
for t in formants_burg.xs():
    for num in range(1, 5):
        formant_value = formants_burg.get_value_at_time(formant_number=num, time=t, unit='HERTZ')
        formants[num].append(formant_value)
df = pd.DataFrame(formants, index=formants_burg.xs())

# フォルマントをプロットする
draw_formants(df)
