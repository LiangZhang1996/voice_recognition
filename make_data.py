import librosa
import numpy as np   
import csv
import sklearn



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

# read in the 'urbansound8K.csv', prepare for datasets
def read_data_list(fname):
    data_list = []
    with open(fname) as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            data_list.append(row)
    return data_list

# read these waves and make them datasets
def get_dataset():
    data_list = read_data_list('urbansound8K/UrbanSound8K.csv')
    X, Y = [], []
    for i in range(1, len(data_list)):
        fname = 'urbansound8K/'+ 'fold' + data_list[i][5] + '/' + data_list[i][0]
        ## MFCC() takes much time
        X.append(MFCC(fname))
        y = int(data_list[i][6])
        Y.append(np.eye(10)[y])
        if i%50==0:
            print(i)
        if i%1000==0:
            ## save the data as .npz file
            np.savez('dataX'+str(i)+'.npz',arr= X)
            print('one file saved..............')
            X = []
    return X,Y

if __name__ == '__main__':
    ## this part, we read wav files and process them with MFCC()
    ## and save the results in these .npz files
    ## it is easy to handle
    X,Y = get_dataset()
    np.savez('dataX9000.npz', arr=X)
    np.savez('dataY.npz', arr=Y)
    ## we may save all these files in ./data , it is easy to read
