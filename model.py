import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import BatchNormalization,Multiply
from tensorflow.keras.models import Sequential, Model

def helper(input_dim,output_dim,input_length,weights):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim,output_dim=output_dim,input_length=input_length,weights=[weights],trainable=False))
    model.add(Bidirectional(LSTM(128, activation = 'tanh', return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences = True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, activation = 'tanh')))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation='tanh'))
    return model


def QuestionPair(input_dim = 100,output_dim = 100,input_length = 100,weights = None,name = None):
    mq1 = helper(input_dim,output_dim,input_length,weights)
    mq2 = helper(input_dim,output_dim,input_length,weights)
    x = Multiply()([mq1.output,mq2.output])
    x = Flatten()(x)
    x = Dense(16,activation="relu")(x)
    x = Dense(2,activation="softmax")(x)
    return Model(inputs = [mq1.input,mq2.input], outputs = x, name = name)


if __name__ == "__main__":
    model = QuestionPair(input_dim=100,output_dim=100,input_length=100,weights=None,name="PairNet")
    print(model.summary())
