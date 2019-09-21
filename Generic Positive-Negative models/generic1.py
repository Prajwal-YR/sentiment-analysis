#import sys
#sys.path.append('c:/users/aksha/appdata/local/programs/python/python37/lib/site-packages')
import pandas as pd  
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
#%%
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner_updated(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()
#%%
#The above functionalities are executed on 'training.1600000.processed.noemoticon' and stored in clean_tweets.csv        
csv = 'clean_tweet.csv'
my_df = pd.read_csv(csv,index_col=0)
my_df.dropna(inplace=True)
my_df.reset_index(drop=True,inplace=True)
#%%
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#%%
X = []
y = []
for index,row in my_df.iterrows():
    X.append([w for w in my_df.iloc[index].text.lower().split() if w not in stop_words])
    y.append(my_df.iloc[index].target)
    if index%10000 == 0:
        print(index)
#%%
import gensim
word_model = gensim.models.Word2Vec(X, size=200, min_count = 10, window = 5)
#%%
pretrained_weights = word_model.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape
print('Result embedding shape:', pretrained_weights.shape)
print('Checking similar words:')
for word in ['stress', 'depressed', 'happy', 'sad']:
  most_similar = ', '.join('%s (%.2f)' % (similar, dist) 
                           for similar, dist in word_model.most_similar(word)[:8])
  print('  %s -> %s' % (word, most_similar))
#%%
def word2idx(word):  
  return word_model.wv.vocab[word].index
X_mod = []
for sentence in X:
    words = filter(lambda x: x in word_model.wv.vocab, sentence)
    t = []
    for word in words:
        t.append(word2idx(word))
    X_mod.append(t)
#%%    
lengths = []
for i in X_mod:
    lengths.append(len(i))
pd.Series(lengths).describe()
#%%
from keras.preprocessing.sequence import pad_sequences
X_pad = pad_sequences(X_mod, maxlen=9, padding='post')
y_in = np.array(y)
y_inp = np.where(y_in==0,0,1)
print(X_pad.shape)
y_inp = y_inp.reshape((len(y_inp),1))
print(y_inp.shape)
#%%
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_pad, y_inp, 
                                                    test_size=0.25, 
                                                    random_state=0)
#%%
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
#%%
#IGNORE for now. Still working on an CNN-LSTM hybrid model
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
epochs, batch_size = 15, 64
n_timesteps, n_features, n_outputs = Xtrain.shape[1], Xtrain.shape[2], ytrain.shape[1]
n_steps, n_length = 4, 2
Xtrain = Xtrain.reshape((Xtrain.shape[0], n_steps, n_length, n_features))
Xtest = Xtest.reshape((Xtest.shape[0], n_steps, n_length, n_features))
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu'), input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size)
'''
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
#%%
model_ = Sequential()
model_.add(Embedding(vocab_size,emdedding_size,input_length=9))
model_.add(LSTM(128))
#model_.add(Dense(50, activation='relu'))
model_.add(Dense(1, activation='sigmoid'))
print(model_.summary())
#%%
model_.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_.fit(Xtrain, ytrain, epochs=5, batch_size=256)
#%%
loss, accuracy = model_.evaluate(Xtest, ytest, verbose=0)
print('Accuracy: %f' % (accuracy))
#%%
INPUT = 'I am fed up with these tests'
test = [w for w in INPUT.lower().split() if w not in stop_words]
words = filter(lambda x: x in word_model.wv.vocab, test)
t = []
for word in words:
    t.append(word2idx(word))
pad = pad_sequences([t], maxlen=9, padding='post')
yhat = model_.predict_classes(np.array(pad))
if(yhat[0] == 0):
    print("Negative")
else:
    print("Positive")
#%%