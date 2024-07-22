import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# WAVファイルのパス
wav_file = "waves/result/test_wave.wav"

# LPC次数
order = 12

# フレーム長とシフト長
frame_length = 512
frame_shift = 256

# WAVファイルを読み込む
waveform, sample_rate = sf.read(wav_file)

# フレームに分割
num_frames = int(np.ceil(len(waveform) / frame_shift))
padded_waveform_length = num_frames * frame_shift + frame_length - frame_shift
padded_waveform = np.pad(waveform, (0, padded_waveform_length - len(waveform)), mode='constant')
frames = np.stack([padded_waveform[i * frame_shift:i * frame_shift + frame_length] for i in range(num_frames)])

# LPC係数の推定
lpc_coeffs = np.zeros((num_frames, order + 1))
for i in range(num_frames):
    frame = frames[i]
    r = np.correlate(frame, frame, mode='full')[frame_length - 1:]
    R = np.zeros((order, order))
    for j in range(order):
        R[j, :] = r[j:j + order]
    r_vector = r[1:order + 1]
    lpc_coeffs[i, 1:] = np.linalg.solve(R, r_vector)
    lpc_coeffs[i, 0] = 1.0

# LPC係数を使用してフォルマント周波数を計算
formants = []
for i in range(num_frames):
    roots = np.roots(lpc_coeffs[i])
    roots = roots[np.imag(roots) >= 0]
    freqs = np.arctan2(np.imag(roots), np.real(roots)) * (sample_rate / (2 * np.pi))
    freqs = freqs[freqs < (sample_rate / 2)]
    formants.append(freqs)

# フォルマント周波数の表示
for i, f in enumerate(formants):
    print("Frame", i+1, ":", f)

# フォルマント周波数のプロット
plt.figure(figsize=(10, 6))
for i, f in enumerate(formants):
    plt.plot(f, np.ones_like(f) * (i+1), 'o', label="Frame {}".format(i+1))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frame')
plt.title('Formant Frequencies')
plt.grid(True)
plt.legend()
plt.show()
