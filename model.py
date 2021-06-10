import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import BatchNormalization,Multiply

def helper(input_dim,output_dim,input_length):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim,output_dim=output_dim,input_length=input_length))
    model.add(Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, activation = 'tanh')))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(16, activation='tanh'))
    return model


def model(input_dim,output_dim,input_length):
    mq1 = helper(2000,128,30)
    mq2 = helper(2000,128,30)
    model = Multiply()([mq1.output,mq2.output])
    model = Flatten()(model)
    print(model.shape)

model()

