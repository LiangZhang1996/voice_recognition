import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
## tensorflow2 is required

## define the model1 
class Mymodel1(keras.Model):
    def __init__(self):
        super(Mymodel1, self).__init__()

        self.CNN = keras.Sequential([
            ## build the CNN layers
            ## con2d layers
            keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=64, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=96, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=128, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            ## fully conected layers
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
            ## output with size (-1, 10)
        ])

    def call(self, x):
        return self.CNN(x)
## define the model2
class Mymodel2(keras.Model):
    def __init__(self):
        super(Mymodel2, self).__init__()

        self.CNN = keras.Sequential([
            ## build the CNN layers
            ## con2d layers
            keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=64, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=96, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            #keras.layers.Conv2D(filters=128, kernel_size=3,activation='relu',padding='same'),
            #keras.layers.MaxPool2D(),
            ## fully conected layers
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
            ## output with size (-1, 10)
        ])

    def call(self, x):
        return self.CNN(x)
## define the model3
class Mymodel3(keras.Model):
    def __init__(self):
        super(Mymodel3, self).__init__()

        self.CNN = keras.Sequential([
            ## build the CNN layers
            ## con2d layers
            keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(filters=64, kernel_size=3,activation='relu',padding='same'),
            keras.layers.MaxPool2D(),
            #keras.layers.Conv2D(filters=96, kernel_size=3,activation='relu',padding='same'),
            #keras.layers.MaxPool2D(),
            #keras.layers.Conv2D(filters=128, kernel_size=3,activation='relu',padding='same'),
            #keras.layers.MaxPool2D(),
            ## fully conected layers
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
            ## output with size (-1, 10)
        ])

    def call(self, x):
        return self.CNN(x)


## load data from our prepared .npz data
def load_dataset(path):
    X = []
    for i in range(1,10):
        data = np.load(path+'dataX'+ str(i) +'.npz')
        X.append(data['arr'])
    Y = np.load(path+'dataY.npz')
    Y = np.array(Y['arr'])
    X1 = X[0]
    for i in range(1,len(X)):
        X1 = np.concatenate((X1,X[i]))
    return X1,Y


## split dataset, build the model and train 
def main():
    X,Y=load_dataset('data/')
    ## shuffle the datasets, makes train better 
    idx  = np.arange(8732)
    np.random.shuffle(idx)
    ## [0,6986） train  80% ; [6986，8732) test 20%
    Xs,Ys = X[idx],Y[idx]
    trainX, testX = Xs[0:6986],Xs[6986:8732]
    trainY, testY = Ys[0:6986],Ys[6986:8732]
    # reshape the train data
    trainX = trainX.reshape((-1,40,173,1))
    testX = testX.reshape((-1,40,173,1))
    print(trainX.shape, testX.shape)
    ## build the model1
    model1=Mymodel1()
    model1.compile(optimizer=keras.optimizers.Adam(),
                loss = keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    print('The model is training with model1....................)
    model1.fit(trainX, trainY, batch_size=64, epochs=10)
    model1.evaluate(testX,testY)
    ## build the model2
    model2=Mymodel2()
    model2.compile(optimizer=keras.optimizers.Adam(),
                loss = keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    print('The model is training with model2....................)
    model2.fit(trainX, trainY, batch_size=64, epochs=10)
    model2.evaluate(testX,testY)
    ## build the model3
    model3=Mymodel3()
    model3.compile(optimizer=keras.optimizers.Adam(),
                loss = keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    print('The model is training with model3....................)
    model3.fit(trainX, trainY, batch_size=64, epochs=10)
    model3.evaluate(testX,testY)

if __name__ == '__main__':
    main()