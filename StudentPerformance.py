import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
#%%
df = pd.read_csv('math.csv')
df = df.drop(['id','school','G1','G2'],axis=1)
df.head()
#%%
sex = LabelEncoder()
df['sex'] = sex.fit_transform(df['sex'])
address = LabelEncoder() 
df['address'] = address.fit_transform(df['address'])
famsize = LabelEncoder()
df['famsize'] = famsize.fit_transform(df['famsize'])
Pstatus = LabelEncoder()
df['Pstatus'] = Pstatus.fit_transform(df['Pstatus'])
Mjob = LabelEncoder()
df['Mjob'] = Mjob.fit_transform(df['Mjob'])
Fjob = LabelEncoder()
df['Fjob'] = Fjob.fit_transform(df['Fjob'])
reason = LabelEncoder()
df['reason'] = reason.fit_transform(df['reason'])
guardian = LabelEncoder()
df['guardian'] = guardian.fit_transform(df['guardian'])
famsup = LabelEncoder()
df['famsup'] = famsup.fit_transform(df['famsup'])
schoolsup = LabelEncoder()
df['schoolsup'] = schoolsup.fit_transform(df['schoolsup'])
paid = LabelEncoder()
df['paid'] = paid.fit_transform(df['paid'])
activities = LabelEncoder()
df['activities'] = activities.fit_transform(df['activities'])
nursery = LabelEncoder()
df['nursery'] = nursery.fit_transform(df['nursery'])
internet = LabelEncoder()
df['internet'] = internet.fit_transform(df['internet'])
higher = LabelEncoder()
df['higher'] = higher.fit_transform(df['higher'])
romantic = LabelEncoder()
df['romantic'] = romantic.fit_transform(df['romantic'])
#%%
l = [sex,address,famsize,Pstatus,Mjob,Fjob,reason,guardian,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic]
ind = [0,2,3,4,7,8,9,10,14,15,16,17,18,19,20,21]
print(len(l))
print(len(ind))
#%%
y = df['G3'].values
#%%
y_in = []
for i in y:
    y_in.append(i/20)
#%%
print(pd.Series(y_in).value_counts())
#%%
dfTrain = df.drop('G3',axis=1)
X = dfTrain.values
print(np.array(X).shape)
#%%
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_in, 
                                                    test_size=0.1, 
                                                    random_state=0)
#%%
scaler = MinMaxScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
#%%
Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
ytrain = np.array(ytrain).reshape(len(ytrain),1)
ytest = np.array(ytest).reshape(len(ytest),1)
#%%
print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
#%%
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
regressor.fit(Xtrain, ytrain)
#%%
y_pred = regressor.predict(Xtest) 
r2_score(ytest, y_pred)
#%%
INPUT = ['M',19,'U','GT3','A',1,4,'at_home','teacher','reputation','father',1,2,1,'yes','yes','yes','no','yes','yes','yes','no',4,3,2,1,1,4,10]
print(len(INPUT))
c = 0
t=[]
for i in range(len(INPUT)):
    if i in ind:
        t.append(l[c].transform([INPUT[i]])[0])
        c = c+1
    else:
        t.append(INPUT[i])
print(t)
t = scaler.transform([t])
y_pred = regressor.predict(t)
print(y_pred) 
#%%




