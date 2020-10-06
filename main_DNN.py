import tensorflow as tf 
import tensorflow.keras as keras 
import numpy as np 

class Mymodel(keras.Model):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.DNN = keras.Sequential([
            keras.layers.Flatten(),
            ## add these fully conected layers
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),           
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),           
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
    def call(self,x):
        return self.DNN(x)
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


if __name__ == '__main__':
    X,Y=load_dataset('data/')
    print(X.shape, Y.shape)
    ## shuffle the datasets, makes train better 
    idx  = np.arange(8732)
    np.random.shuffle(idx)
    ## [0,6986） train  80% ; [6986，8732) test 20%
    Xs,Ys = X[idx],Y[idx]
    trainX, testX = Xs[0:6986],Xs[6986:8732]
    trainY, testY = Ys[0:6986],Ys[6986:8732]
    print(trainX.shape, trainY.shape)
    print(testX.shape, testY.shape)
    ## build the model
    model=Mymodel()
    model.compile(optimizer=keras.optimizers.Adam(),
                loss = keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    ## fit the model
    print('Training the model.......................')
    model.fit(trainX, trainY, batch_size=64, epochs=10)
    ## test the model
    model.evaluate(testX,testY)