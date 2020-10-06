# MFCC特征提取
## 1 处理音频
针对特定的音频有特定处理，我们使用的是UrbanSound8K数据集，每个音频几乎都是4s，采样率22050， 4s一共88200个采样点。因此读取后的音频填充0到大小一致
```swift
x, sr = librosa.load(fname)
## 22050* 4s
if x.shape[0]< 88200:
    x = np.pad(x,(0, 88200-x.shape[0]),'constant')
x = x[0:88200]
```
下面使用 librosa的函数提取MFCC特征
```swift
## mfccs 提取mfccs特征
mfccs = librosa.feature.mfcc(y=x, sr=sr,n_mfcc=40)
## normalize the MFCCs  标准化
norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
```
在`librosa.feature.mfcc(y=x, sr=sr,n_mfcc=40)`中，我们得到 （40* 173）的MFCCs. `n_mfcc=40`说明返回MFCCs特征的数量。更多问题参考[MFCC函数介绍](http://librosa.github.io/librosa/generated/librosa.feature.mfcc.html)。
这就是MFCC特征的提取过程。

# CNN结构介绍

下面是用于定义`CNN`的代码
```swift
self.CNN = keras.Sequential([
        ## build the CNN layers
        keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu',padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=64, kernel_size=3,activation='relu',padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=96, kernel_size=3,activation='relu',padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(filters=128, kernel_size=3,activation='relu',padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
        ])
```
详细说明如下图
![](net2.png)