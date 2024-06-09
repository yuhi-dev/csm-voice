#!/usr/bin/env python
# -*- coding:utf-8 -*-

import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import argparse
import os

sns.set()
plt.rcParams['figure.dpi'] = 100

# 現在のタイムスタンプを取得する関数
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 音声ファイルを読み込む関数
def load_sound(file_path):
    return parselmouth.Sound(file_path)

# 音声の波形をプロットする関数
def plot_waveform(sound):
    plt.figure()
    plt.plot(sound.xs(), sound.values.T)
    plt.xlim([sound.xmin, sound.xmax])
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    timestamp = get_timestamp()
    plt.savefig(f"waveform_{timestamp}.png")
    plt.show()

# スペクトログラムを描画する関数
def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

# インテンシティ（音の強さ）を描画する関数
def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

# スペクトログラムとインテンシティを同時にプロットする関数
def plot_spectrogram_with_intensity(sound):
    intensity = sound.to_intensity()
    spectrogram = sound.to_spectrogram()
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_intensity(intensity)
    plt.xlim([sound.xmin, sound.xmax])
    timestamp = get_timestamp()
    plt.savefig(f"spectrogram_intensity_{timestamp}.png")
    plt.show()

# ピッチ（基本周波数）を描画する関数
def draw_pitch(pitch):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

# スペクトログラムとピッチを同時にプロットする関数
def plot_spectrogram_with_pitch(sound):
    pitch = sound.to_pitch()
    pre_emphasized_snd = sound.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)
    plt.figure()
    draw_spectrogram(spectrogram)
    plt.twinx()
    draw_pitch(pitch)
    plt.xlim([sound.xmin, sound.xmax])
    timestamp = get_timestamp()
    plt.savefig(f"spectrogram_pitch_{timestamp}.png")
    plt.show()

# フォルマント（共鳴周波数）を描画する関数
def draw_formants(formants, representative_values, filename):
    for num in range(1, 5):
        plt.plot(formants.index, formants[num], marker='o', linestyle='-', markersize=4, label=f"F{num}")
    
    # 代表的な値をプロット
    for num, value in representative_values.items():
        plt.hlines(value, formants.index[0], formants.index[-1], linestyles='dashed', label=f"{num}: {value:.2f} Hz")
    
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Formants")
    plt.legend()
    timestamp = get_timestamp()
    plt.savefig(f"{filename}_{timestamp}.png")
    plt.show()

# フォルマントを抽出する関数
def extract_formants(sound):
    formants_burg = sound.to_formant_burg(max_number_of_formants=5.0, maximum_formant=12000.0, window_length=0.025, pre_emphasis_from=50.0)
    formants = {i: [] for i in range(1, 5)}

    for t in formants_burg.xs():
        for num in range(1, 5):
            formant_value = formants_burg.get_value_at_time(formant_number=num, time=t, unit='HERTZ')
            formants[num].append(formant_value)
    df = pd.DataFrame(formants, index=formants_burg.xs())

    # 各フォルマントの代表的な周波数値を出力する
    representative_values = {}
    for num in range(1, 5):
        median_value = df[num].median()
        representative_values[f"Formant_{num}"] = median_value
    
    return df, representative_values

# フォルマントをプロットする関数
def plot_formants(df, representative_values, filename):
    draw_formants(df, representative_values, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and plot audio features from a sound file.")
    parser.add_argument('file_path', type=str, help='Path to the audio file to be analyzed')
    args = parser.parse_args()

    snd = load_sound(args.file_path)

    filename = os.path.splitext(os.path.basename(args.file_path))[0]
    
    # plot_waveform(snd)
    
    # plot_spectrogram_with_intensity(snd)
    
    # plot_spectrogram_with_pitch(snd)
    
    formants_df, representative_values = extract_formants(snd)
    plot_formants(formants_df, representative_values, filename)

    print(representative_values)

