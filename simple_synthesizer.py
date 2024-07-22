import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# サンプリング周波数
fs = 44100

# 各母音の周波数
vowel_frequencies = {
    'formant0': 260,  
    'formant1': 520,  
    'formant2': 780,  
    'formant3': 1040,  
    'formant4': 0
}

# 各母音の長さ（秒）
vowel_durations = {
    'formant0': 3.0,
    'formant1': 3.0,
    'formant2': 3.0,
    'formant3': 3.0,
    'formant4': 3.0
}

# 各母音の振幅
amplitudes = {
    'formant0': 0.18,
    'formant1': 0.14,
    'formant2': 0.13,
    'formant3': 0.03,
    'formant4': 0.00
}

def generate_waveform(duration, frequency, amplitude=1.0, waveform_type='sine'):
    """
    指定された周波数と持続時間に基づいて波形を生成

    パラメータ:
    - duration (float): 波形の持続時間（秒）
    - frequency (float): 波形の周波数（ヘルツ）
    - amplitude (float): 波形の振幅
    - waveform_type (str): 使用する波形の種類。'sine'、'triangle'、'square'、'sawtooth'のいずれか

    戻り値:
    - waveform (numpy.ndarray): 生成された波形を含む配列
    """

    # 0からdurationまでの時間値を等間隔に生成
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 指定された波形タイプに基づいて波形を生成
    if waveform_type == 'sine':
        waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    elif waveform_type == 'triangle':
        waveform = amplitude * (2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi)
    elif waveform_type == 'square':
        waveform = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform_type == 'sawtooth':
        waveform = amplitude * (2 * (t * frequency - np.floor(t * frequency + 0.5)))

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
    # フォルマントを合成してWaveファイルに保存する
    vowel_sequence = ['formant0', 'formant1', 'formant2', 'formant3', 'formant4']

    # 合成波形を入れるための空配列を作っておく
    combined_sound = np.array([])

    # 合成波形を作る
    for vowel in vowel_sequence:
        frequency = vowel_frequencies[vowel]
        duration = vowel_durations[vowel]
        amplitude = amplitudes[vowel]  # amplitudeを取得
        vowel_sound = generate_waveform(duration, frequency, amplitude=amplitude, waveform_type='square')
        # 重ね合わせる
        if combined_sound.size == 0:
            combined_sound = vowel_sound
        else:
            # 波形を重ね合わせる
            combined_sound += vowel_sound

    # 波形を正規化する（[-1, 1]の範囲に収めるとWaveファイル化できるため）
    normalized_wave = np.int16(combined_sound / np.max(np.abs(combined_sound)) * 32767)

    # WAVファイルとして保存する
    write('waves/result/o_test_wave.wav', fs, normalized_wave)
