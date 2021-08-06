import string
import re
import nltk
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from model import QuestionPair
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split as tts

print("[INFO] All the libraries are imported....")

data = pd.read_csv('train.csv')

print("[INFO] Dataset loaded....")

# preprocessing the data
def cleaning(text):
    text = text.lower()
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean,'',text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text

print("[INFO] Cleaning is taking place....")

data['question1'] = data['question1'].map(cleaning)
data['question2'] = data['question2'].map(cleaning)

print("[INFO] Dataset cleaning is over....")

x = data[['question1','question2']].values
y = data['is_duplicate'].values

y = to_categorical(y)

xtrain, xtest, ytrain, ytest = tts(x, y,test_size=0.15,random_state=123,stratify=y)
print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)


# converting to text to sequences
tokenizer = Tokenizer(5000,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(xtrain)
xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = pad_sequences(xtrain,maxlen=100,padding='post')
xtest = pad_sequences(xtest,maxlen=100,padding='post')
print("[INFO] Data preprocessing is over....")

# fitting it into the data
print("[INFO] Running the model....")
hist = model.fit(xtrain,ytrain,epochs=20,validation_data=(xtest,ytest))

print("[INFO] Saving the model into the disk....")
model.save('reviews.h5')
print("[INFO] Model saved into the disk....")


# plotting the figures
print("[INFO] Plotting the figures....")
plt.figure(figsize=(15,10))
plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.title("Model Accuracy vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.legend(loc='lower right')
plt.savefig('./img/accuracy.png')


plt.figure(figsize=(15,10))
plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.title("Model Loss vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.legend(loc='upper right')
plt.savefig('./img/loss.png')
print("[INFO] Figures saved in the disk....")

# testing the model
print("[INFO] Testing the model....")
print("[INFO] The result obtained is...\n")
model.evaluate(xtest,ytest)

joblib.dump(tokenizer,'tokenizer.pkl')