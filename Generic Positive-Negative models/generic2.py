#import sys
#sys.path.append('c:/users/aksha/appdata/local/programs/python/python37/lib/site-packages')
import pandas as pd  
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import LSTM
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
X = []
y = []
for index,row in my_df.iterrows():
    X.append([w for w in my_df.iloc[index].text.lower().split()])
    y.append(my_df.iloc[index].target)
    if index%10000 == 0:
        print(index)
#%%
X = np.array(X) 
print(X.shape)
y = np.array(y).reshape((len(y),1))
print(y.shape)
#%%
x = my_df['text']
y = my_df['target'] 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=13)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#%%
tk = Tokenizer(num_words=30000,
filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
tk.fit_on_texts(X_train)
X_train_seq = tk.texts_to_sequences(X_train)
X_test_seq = tk.texts_to_sequences(X_test)
#%%
lengths = []
for i in X_train_seq:
    lengths.append(len(i))
pd.Series(lengths).describe()
#%%
X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=16)
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=16)
#%%
ytrain = np.where(y_train==0,0,1)
ytest = np.where(y_test==0,0,1)
#%%
model = Sequential()
model.add(Embedding(30000, 8, input_length=16))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_seq_trunc, ytrain, epochs=4, batch_size=512)
_, accuracy = model.evaluate(X_test_seq_trunc, ytest)
print(accuracy)
#%%
INPUT = 'I am very happy with this service'
t = tk.texts_to_sequences([INPUT])
pad = pad_sequences(t, maxlen=16, padding='post')
yhat = model.predict_classes(np.array(pad))
if(yhat[0] == 0):
    print("Negative")
else:
    print("Positive")
#%%