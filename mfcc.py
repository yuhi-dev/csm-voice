import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.signal
import librosa
import os

# 高域強調
def preEmphasis(wave, p=0.97):
    return scipy.signal.lfilter([1.0, -p], 1, wave)

# MFCCの計算
def mfcc(wave, fs, label):
    mfccs = librosa.feature.mfcc(wave, sr=fs, n_fft=512)
    mfccs = np.average(mfccs, axis=1)
    mfccs = mfccs.flatten().tolist()
    # MFCCの第1次元と14次元以降の特徴を削除
    mfccs.pop(0)
    mfccs = mfccs[:12]
    mfccs.insert(0, label)
    return mfccs

# データの読み込み, 音素毎にMFCCを計算(使用データは500ファイル分)
mfcc_data = []
boin_list = ["a", "i", "u", "e", "o"]
directory = "waves/"

for filename in os.listdir(directory):
    if filename.endswith(".lab"):
        open_file = os.path.join(directory, filename)
        filename_without_ext = os.path.splitext(filename)[0]
        audio_file = os.path.join(directory, filename_without_ext)
        v, fs = librosa.load(audio_file + ".wav", sr=None)
        with open(open_file, "r") as f:
            data_list = [line.split() for line in f.readlines()]
            for data in data_list:
                label = data[2]
                if label in boin_list:
                    start = int(fs * float(data[0]))
                    end = int(fs * float(data[1]))
                    voice_data = v[start:end]
                    if end - start <= 512:
                        continue
                    hammingWindow = np.hamming(len(voice_data))
                    voice_data = voice_data * hammingWindow
                    voice_data = preEmphasis(voice_data, p=0.97)
                    mfcc_data.append(mfcc(voice_data, fs, label))

# データセットの読み込み
df = pd.DataFrame(mfcc_data)
x = df.iloc[:, 1:]  # MFCCで得た特徴点
y = df.iloc[:, 0]   # 母音のラベル

# ラベルを数値に変換
label_set = set(y)
label_list = sorted(list(label_set))
label_dict = {label: idx for idx, label in enumerate(label_list)}
y = y.map(label_dict)
y = np.array(y, dtype="int")

# 教師データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

# データの標準化
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# SVMモデルのトレーニング
model_linear = SVC(kernel='linear', random_state=1)
model_poly = SVC(kernel="poly", random_state=1)
model_rbf = SVC(kernel="rbf", random_state=1)

model_linear.fit(x_train_std, y_train)
model_poly.fit(x_train_std, y_train)
model_rbf.fit(x_train_std, y_train)

# トレーニングデータでの性能評価
pred_linear_train = model_linear.predict(x_train_std)
pred_poly_train = model_poly.predict(x_train_std)
pred_rbf_train = model_rbf.predict(x_train_std)
accuracy_linear_train = accuracy_score(y_train, pred_linear_train)
accuracy_poly_train = accuracy_score(y_train, pred_poly_train)
accuracy_rbf_train = accuracy_score(y_train, pred_rbf_train)

print("Training Results:")
print("Linear Kernel Accuracy:", accuracy_linear_train)
print("Polynomial Kernel Accuracy:", accuracy_poly_train)
print("RBF Kernel Accuracy:", accuracy_rbf_train)

# テストデータでの性能評価
pred_linear_test = model_linear.predict(x_test_std)
pred_poly_test = model_poly.predict(x_test_std)
pred_rbf_test = model_rbf.predict(x_test_std)
accuracy_linear_test = accuracy_score(y_test, pred_linear_test)
accuracy_poly_test = accuracy_score(y_test, pred_poly_test)
accuracy_rbf_test = accuracy_score(y_test, pred_rbf_test)

print("-" * 40)
print("Testing Results:")
print("Linear Kernel Accuracy:", accuracy_linear_test)
print("Polynomial Kernel Accuracy:", accuracy_poly_test)
print("RBF Kernel Accuracy:", accuracy_rbf_test)
