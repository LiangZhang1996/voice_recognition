import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import librosa
import sklearn

class Mymodel(keras.Model):
    def __init__(self):
        super(Mymodel, self).__init__()

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

    def call(self, x):
        return self.CNN(x)

# read in the wave and calculate the MFCCs
def MFCC(fname):
    ## read the file and padding to the same 
    x, sr = librosa.load(fname)
    ## 22050* 4s
    if x.shape[0]< 88200:
        x = np.pad(x,(0, 88200-x.shape[0]),'constant')
    x = x[0:88200]
    ## mfccs
    mfccs = librosa.feature.mfcc(y=x, sr=sr,n_mfcc=40)
    ## normalize the MFCCs 
    norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return norm_mfccs
def recong(y):
    y = y.reshape((10,))
    z = np.sort(y)
    label =['airconditioner', 'carhorn', 'childrenplaying', 'dogdark', 'drilling', 'engineidling', 'gunshot', 'jackhammer', 'siren','street_music']
    for yy in [z[-1], z[-2], z[-3]]:
        c = np.where(y==yy)
        print('Recongnized:',label[int(c[0])], 'Probability:',y[int(c[0])])



if __name__ == '__main__':
    ## build the model
    model = Mymodel()
    model.compile(optimizer=keras.optimizers.Adam(),
                loss = keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    ## load the pretrained model weights
    # 'weights/weights' is the weights you have ever saved
    model.load_weights('weights/weights')
    ## input the audio file path+ name and will output the results
    ## you should input the fname 
    mfccs = MFCC(fname) 
    mfccs = mfccs.reshape((-1,40,173,1))
    ## recongnition 
    y  = model.predict(mfccs)
    recong(y)



