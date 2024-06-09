import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# サンプリング周波数
fs = 44100

# 各母音の周波数
vowel_frequencies = {
    'a': 440,  # A4
    'i': 660,  # E5
    'u': 330,  # E4
    'e': 550,  # C#5
    'o': 220   # A3
}

# 各母音の長さ（秒）
vowel_durations = {
    'a': 0.25,
    'i': 0.25,
    'u': 0.25,
    'e': 0.25,
    'o': 0.25
}

def generate_waveform(duration, frequency, waveform_type='sine'):
    """
    指定された周波数と持続時間に基づいて波形を生成

    パラメータ:
    - duration (float): 波形の持続時間（秒）
    - frequency (float): 波形の周波数（ヘルツ）
    - waveform_type (str): 使用する波形の種類。'sine'、'triangle'、'square'、'sawtooth'のいずれか

    戻り値:
    - waveform (numpy.ndarray): 生成された波形を含む配列
    """

    # 0からdurationまでの時間値を等間隔に生成
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 指定された波形タイプに基づいて波形を生成
    if waveform_type == 'sine':
        waveform = np.sin(2 * np.pi * frequency * t)
    elif waveform_type == 'triangle':
        waveform = 2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi
    elif waveform_type == 'square':
        waveform = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform_type == 'sawtooth':
        waveform = 2 * (t * frequency - np.floor(t * frequency + 0.5))

    return waveform


def plot_vowel_sound(vowel_sound):

    # 母音の音声波形を可視化する(デバッグ用)
    plt.figure(figsize=(10, 4))
    plt.plot(vowel_sound, color='blue')
    plt.title('Vowel Sound Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # あいうえおを合成してWaveファイルに保存する
    vowel_sequence = ['a', 'i', 'u', 'e', 'o']

    # 合成波形を入れるための空配列を作っておく
    combined_sound = np.array([])
    
    # 合成波形を作る
    for vowel in vowel_sequence:
        frequency = vowel_frequencies[vowel]
        duration = vowel_durations[vowel]
        vowel_sound = generate_waveform(duration, frequency, waveform_type='triangle')
        plot_vowel_sound(vowel_sound)
        combined_sound = np.concatenate((combined_sound, vowel_sound))

    # 波形を正規化する（[-1, 1]の範囲に収めるとWaveファイル化できるため）
    normalized_wave = np.int16(combined_sound * 32767)

    # WAVファイルとして保存する
    write('aiueo_wave.wav', fs, normalized_wave)
