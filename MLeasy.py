#!/usr/bin/env python
# coding: utf-8

# In[2]:


def LogisticRegression(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    print('Check if you need Normalice X data before apply regression')
    print('A')
    print('if need it: X = preprocessing.StandardScaler().fit(X).transform(X)')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    
    LR = LogisticRegression(C=0.1, solver='liblinear').fit(X_train,y_train)
    y_hat = LR.predict(X_test) #nos dice si es 0 o 1
    print('LR: ')
    print(LR)
    print('*****************')
    print('y_hat: ')
    print(y_hat)
    print('*****************')
    yhat_prob = LR.predict_proba(X_test) #nos dice la probab. de que pertenezca a clase 0 o 1.
    print('yhat probability: ')
    print(yhat_prob)
    print('*****************')
    from sklearn.metrics import classification_report
    print (classification_report(y_test, y_hat))
    print('*****************')
    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(y_test, y_hat)
    print('MSE - Mean Squared Error: ', MSE)
    from sklearn.metrics import jaccard_score
    Jaccard = jaccard_score(y_test, y_hat)
    print('Jaccard: ', Jaccard)
    from sklearn.metrics import recall_score
    Recall = recall_score(y_test, y_hat)
    print('Recall: ', Recall)
    from sklearn.metrics import balanced_accuracy_score
    BAS = balanced_accuracy_score(y_test, y_hat)
    print('BAS - Balanced Accuracy Score: ', BAS)
    from sklearn.metrics import average_precision_score
    APS = average_precision_score(y_test, y_hat)
    print('APS - Average Precission Score: ', APS)
    from sklearn.metrics import log_loss
    Logloss = log_loss(y_test, y_hat)
    print('Logloss: ', Logloss)
    clf = LogisticRegression(random_state=0).fit(X, y)
    clf.predict(X)
    #clf.predict_proba(X[:2, :])
    print('Score: ',clf.score(X, y))
    print('*****************')


# In[1]:


def LogisticRegressionCode(df):
    print("""
print('Check if you need Normalice X data before apply regression')
print('if need it: X = preprocessing.StandardScaler().fit(X).transform(X)')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', Xtrain.shape, ytrain.shape)
print('Test size: ', Xtest.shape, ytest.shape)
    
LR = LogisticRegression(C=0.1, solver='liblinear').fit(Xtrain,ytrain)
y_hat = LR.predict(Xtest) #nos dice si es 0 o 1
print('LR: ')
print(LR)
print('*****************')
print('y_hat: ')
print(y_hat)
print('*****************')
yhat_prob = LR.predict_proba(Xtest) #nos dice la probab. de que pertenezca a clase 0 o 1.
print('yhat probability: ')
print(yhat_prob)
print('*****************')
from sklearn.metrics import classification_report
print (classification_report(ytest, y_hat))
print('*****************')
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(ytest, y_hat)
print('MSE - Mean Squared Error: ', MSE)
from sklearn.metrics import jaccard_score
Jaccard = jaccard_score(ytest, y_hat)
print('Jaccard: ', Jaccard)
from sklearn.metrics import recall_score
Recall = recall_score(ytest, y_hat)
print('Recall: ', Recall)
from sklearn.metrics import balanced_accuracy_score
BAS = balanced_accuracy_score(ytest, y_hat)
print('BAS - Balanced Accuracy Score: ', BAS)
from sklearn.metrics import average_precision_score
APS = average_precision_score(ytest, y_hat)
print('APS - Average Precission Score: ', APS)
from sklearn.metrics import log_loss
Logloss = log_loss(ytest, y_hat)
print('Logloss: ', Logloss)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X)
#clf.predict_proba(X[:2, :])
print('Score: ',clf.score(X, y))
print('*****************')
    """)


# In[1]:


def normalize(x_df):
    ## Normalizar
    from sklearn import preprocessing

    x_df = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_df_scaled = min_max_scaler.fit_transform(x_df)
    df = pd.DataFrame(x_df_scaled)
    return df


# In[6]:


def normalizeCode(x_df):
    print("""
## Normalizar
from sklearn import preprocessing

x_df = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_df_scaled = min_max_scaler.fit_transform(x_df)
df = pd.DataFrame(x_df_scaled)
return df
    """)


# In[7]:


def imputation(df):
    ## La imputacion se hace para reemplazar los nans por un valor estimado

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputer.fit(df)
    df_num_imp = imputer.transform(df)
    df = pd.DataFrame(df_num_imp, columns = df.columns)
    return df


# In[5]:


def imputationCode(df):
    print("""
## La imputacion se hace para reemplazar los nans por un valor estimado

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
imputer.fit(df)
df_num_imp = imputer.transform(df)
df = pd.DataFrame(df_num_imp, columns = df.columns)
return df
    """)


# In[ ]:


def tonumeric(df):
    for i in df.columns:
        keys = df[i].unique()
        values = range(len(keys))
        dictionary = dict(zip(keys, values))
        df[i] = df[i].replace(dictionary)
    return df


# In[8]:


def tonumericCode(df):
    print("""
for i in df.columns:
    keys = df[i].unique()
    values = range(len(keys))
    dictionary = dict(zip(keys, values))
    df[i] = df[i].replace(dictionary)
return df
    """)


# In[2]:


def KNN(df):
    print('¡¡¡ATTENTION!!!: KNN could take long time')
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import neighbors
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    k = int(input('write numbers of k:'))
    w = input('write kind of weights, distance(most usual) or uniform: ')
    clf = neighbors.KNeighborsClassifier(k, weights=w)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    print('yhat: ')
    print(yhat)
    print('********')
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average="macro"))
    print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
    print("F1-score: ", f1_score(y_test,yhat, average="macro"))


# In[1]:


def KNNCode(df):
    print("""
print('¡¡¡ATTENTION!!!: KNN could take long time')
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
k = int(input('write numbers of k:'))
w = input('write kind of weights, distance(most usual) or uniform: ')
clf = neighbors.KNeighborsClassifier(k, weights=w)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print('yhat: ')
print(yhat)
print('********')
print("Exactitud: " , accuracy_score(y_test, yhat))
print("Precisión: ", precision_score(y_test, yhat, average="macro"))
print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
print("F1-score: ", f1_score(y_test,yhat, average="macro"))
    """)


# In[1]:


def NB(df):
    print('¡¡¡ATTENTION!!!: KNN could take long time')
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    yhat = gnb.predict(X_test)
    print('yhat: ')
    print(yhat)
    print('********')
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average="macro"))
    print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
    print("F1-score: ", f1_score(y_test,yhat, average="macro"))


# In[3]:


def NBCode(df):
    print("""
print('¡¡¡ATTENTION!!!: KNN could take long time')
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
yhat = gnb.predict(X_test)
print('yhat: ')
print(yhat)
print('********')
print("Exactitud: " , accuracy_score(y_test, yhat))
print("Precisión: ", precision_score(y_test, yhat, average="macro"))
print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
print("F1-score: ", f1_score(y_test,yhat, average="macro"))
    """)


# In[4]:


def SVM(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    krn = input('Choose a kernel: linear, rbf (exponential), poly (polynomial), sigmoid or precomputed:')
    clf = svm.SVC(kernel=krn)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    print('yhat: ')
    print(yhat)
    print('********')
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average="macro"))
    print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
    print("F1-score: ", f1_score(y_test,yhat, average="macro"))


# In[5]:


def SVMCode(df):
    print("""
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
krn = input('Choose a kernel: linear, rbf (exponential), poly (polynomial), sigmoid or precomputed:')
clf = svm.SVC(kernel=krn)
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print('yhat: ')
print(yhat)
print('********')
print("Exactitud: " , accuracy_score(y_test, yhat))
print("Precisión: ", precision_score(y_test, yhat, average="macro"))
print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
print("F1-score: ", f1_score(y_test,yhat, average="macro"))
    """)


# In[6]:


def SVMAll(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    krn = ['linear', 'rbf', 'poly', 'sigmoid']
    for i in krn:
        clf = svm.SVC(kernel=i)
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)
        print('Para', i, ':')
        print("Exactitud: " , accuracy_score(y_test, yhat))
        print("Precisión: ", precision_score(y_test, yhat, average="macro"))
        print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
        print("F1-score: ", f1_score(y_test,yhat, average="macro"))
        print('*****************************')


# In[7]:


def SVMAllCode(df):
    print("""
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
krn = ['linear', 'rbf', 'poly', 'sigmoid']
for i in krn:
    clf = svm.SVC(kernel=i)
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    print('Para', i, ':')
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average="macro"))
    print("Sensibilidad: ", recall_score(y_test, yhat, average="macro"))
    print("F1-score: ", f1_score(y_test,yhat, average="macro"))
    print('*****************************')
    """)


# In[9]:


def Euclidean(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestCentroid
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
    clf = NearestCentroid()
    clf.fit(X, y)
    yhat = clf.predict(X_test)
    print('yhat: ')
    print(yhat)
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average=avg))
    print("Sensibilidad: ", recall_score(y_test, yhat, average=avg))
    print("F1-score: ", f1_score(y_test,yhat, average=avg))


# In[10]:


def EuclideanCode(df):
    print("""
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
clf = NearestCentroid()
clf.fit(X, y)
yhat = clf.predict(X_test)
print('yhat: ')
print(yhat)
print("Exactitud: " , accuracy_score(y_test, yhat))
print("Precisión: ", precision_score(y_test, yhat, average=avg))
print("Sensibilidad: ", recall_score(y_test, yhat, average=avg))
print("F1-score: ", f1_score(y_test,yhat, average=avg))
    """)


# In[11]:


def DecisionTree(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
    crit = input('Choose a criterion for Decision Tree: gini(default) or entropy:')
    clf = tree.DecisionTreeClassifier(criterion=crit)
    clf = clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    print('yhat: ')
    print(yhat)
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average=avg))
    print("Sensibilidad: ", recall_score(y_test, yhat, average=avg))
    print("F1-score: ", f1_score(y_test,yhat, average=avg))


# In[12]:


def DecisionTreeCode(df):
    print("""
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
crit = input('Choose a criterion for Decision Tree: gini(default) or entropy:')
clf = tree.DecisionTreeClassifier(criterion=crit)
clf = clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print('yhat: ')
print(yhat)
print("Exactitud: " , accuracy_score(y_test, yhat))
print("Precisión: ", precision_score(y_test, yhat, average=avg))
print("Sensibilidad: ", recall_score(y_test, yhat, average=avg))
print("F1-score: ", f1_score(y_test,yhat, average=avg))
    """)


# In[17]:


def RF_GridSearch(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    print('¡¡¡ATTETION!!!: This classifier includes GridSearch')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    clf = RandomForestClassifier()
    C2 = input('Enter # of features to considerer in each split (list format: 2,3,5,10... / Recomended 2,5):')
    C2 = C2.split(',')
    C2 = [int(i) for i in C2]
    C3 = input('Enter # of Max Leaf Nodes in each Tree (list format: 2,3,5,10... / Recomended 3,5):')
    C3 = C3.split(',')
    C3 = [int(i) for i in C3]
    C4 = input('Enter Max Depth in each Tree (list format: 2,3,5,10... / Recomended 4,8):')
    C4 = C4.split(',')
    C4 = [int(i) for i in C4]
    C6 = input('Enter Min of samples split (list format: 2,3,5,10... / Recommended 3,5):')
    C6 = C6.split(',')
    C6 = [int(i) for i in C6]
    C7 = input('Enter # of estimators (list format: 50,100,250,1000... / Recommended 2 or 3 values):')
    C7 = C7.split(',')
    C7 = [int(i) for i in C7]
    params={'criterion':['gini', 'entropy'],
        'max_depth': C4,# Maxima pofundidad del arbol
        'max_features': C2, # numero de features a considerar en cada split
        'max_leaf_nodes': C3, # maximo de nodos del arbol
        'min_impurity_decrease' : [0.02], # un nuevo nodo se hará si al hacerse se decrece la impureza
        'min_samples_split': C6, # The minimum number of samples required to split an internal node
        'n_estimators':C7
        }
    grid_solver = GridSearchCV(estimator = clf, # model to train
                   param_grid = params, # param_grid
                   scoring = make_scorer(f1_score, average ="macro"),
                   cv = 5)
    model_result = grid_solver.fit(X,y)
    model_result.best_estimator_
    model_result.best_score_
    model_result.best_params_
    print(model_result.best_estimator_)
    print(model_result.best_score_)
    print(model_result.best_params_ )


# In[4]:


def RF_GridSearchCode(df):
    print("""
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
clf = RandomForestClassifier()
C2 = input('Enter # of features to considerer in each split (list format: 2,3,5,10... / Recomended 2,5):')
C2 = C2.split(',')
C2 = [int(i) for i in C2]
C3 = input('Enter # of Max Leaf Nodes in each Tree (list format: 2,3,5,10... / Recomended 3,5):')
C3 = C3.split(',')
C3 = [int(i) for i in C3]
C4 = input('Enter Max Depth in each Tree (list format: 2,3,5,10... / Recomended 4,8):')
C4 = C4.split(',')
C4 = [int(i) for i in C4]
C6 = input('Enter Min of samples split (list format: 2,3,5,10... / Recommended 3,5):')
C6 = C6.split(',')
C6 = [int(i) for i in C6]
C7 = input('Enter # of estimators (list format: 50,100,250,1000... / Recommended 2 or 3 values):')
C7 = C7.split(',')
C7 = [int(i) for i in C7]
params={'criterion':['gini', 'entropy'],
    'max_depth': C4,# Maxima pofundidad del arbol
    'max_features': C2, # numero de features a considerar en cada split
    'max_leaf_nodes': C3, # maximo de nodos del arbol
    'min_impurity_decrease' : [0.02], # un nuevo nodo se hará si al hacerse se decrece la impureza
    'min_samples_split': C6, # The minimum number of samples required to split an internal node
    'n_estimators':C7
        }
grid_solver = GridSearchCV(estimator = clf, # model to train
                param_grid = params, # param_grid
                scoring = make_scorer(f1_score, average ="macro"),
                cv = 5)
model_result = grid_solver.fit(X,y)
model_result.best_estimator_
model_result.best_score_
model_result.best_params_
print(model_result.best_estimator_)
print(model_result.best_score_)
print(model_result.best_params_ )
    """)


# In[5]:


def RF(df):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    yhat = []
    C1 = input('Criterion (gini or entropy):')
    C2 = int(input('Enter # of features to considerer in each split (list format: 2,3,5,10... / Recomended 2,5):'))
    C3 = int(input('Enter # of Max Leaf Nodes in each Tree (list format: 2,3,5,10... / Recomended 3,5):'))
    C4 = input('Enter Max Depth in each Tree (list format: 2,3,5,10... / Recomended 4,8):')
    C6 = int(input('Enter Min of samples split (list format: 2,3,5,10... / Recommended 3,5):'))
    C7 = int(input('Enter # of estimators (list format: 50,100,250,1000... / Recommended 2 or 3 values):')) 
    avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
    clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=4, min_samples_split=3, max_features=2, max_leaf_nodes=8, min_impurity_decrease=0.02)
    clf.fit(X_train, y_train)  ##X_train, y_train
    yhat1 = clf.predict(X_test)
    yhat.append(yhat1)
    print('yhat:')
    print(yhat)
    print('************')
    print("Exactitud: " , accuracy_score(y_test, yhat))
    print("Precisión: ", precision_score(y_test, yhat, average=avg))
    print("Sensibilidad: ", recall_score(y_test, yhat, average=avg))
    print("F1-score: ", f1_score(y_test,yhat, average=avg))


# In[6]:


def RF(df):
    print("""
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
yhat = []
C1 = input('Criterion (gini or entropy):')
C2 = int(input('Enter # of features to considerer in each split (list format: 2,3,5,10... / Recomended 2,5):'))
C3 = int(input('Enter # of Max Leaf Nodes in each Tree (list format: 2,3,5,10... / Recomended 3,5):'))
C4 = input('Enter Max Depth in each Tree (list format: 2,3,5,10... / Recomended 4,8):')
C6 = int(input('Enter Min of samples split (list format: 2,3,5,10... / Recommended 3,5):'))
C7 = int(input('Enter # of estimators (list format: 50,100,250,1000... / Recommended 2 or 3 values):')) 
avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
clf = RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=4, min_samples_split=3, max_features=2, max_leaf_nodes=8, min_impurity_decrease=0.02)
clf.fit(X_train, y_train)  ##X_train, y_train
yhat1 = clf.predict(X_test)
yhat.append(yhat1)
print('yhat:')
print(yhat)
print('************')
print("Exactitud: " , accuracy_score(y_test, yhat))
print("Precisión: ", precision_score(y_test, yhat, average=avg))
print("Sensibilidad: ", recall_score(y_test, yhat, average=avg))
print("F1-score: ", f1_score(y_test,yhat, average=avg))
    """)


# In[1]:


def AdaBoost(df):
    print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import sklearn
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    classname = input('Name of the class column:')
    ts = float(input('write test size: '))
    rs = int(input("write random_state(default 0, most common 4 or 2): "))
    y = np.asarray(df[classname])
    X = np.asarray(df.loc[:, df.columns != classname])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
    print('Train size: ', X_train.shape, y_train.shape)
    print('Test size: ', X_test.shape, y_test.shape)
    print('********')
    avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
    NE = int(input('Number of Estimators:'))
    CV = int(input('cv or cross-validation generator or an iterable / optional / Most frequent 5 or 10:'))
    yhat=[]
    clf = AdaBoostClassifier(n_estimators=NE)
    clf.fit(X, y)
    yhat1 = clf.predict(X_test)
    yhat.append(yhat1)
    print('yhat:')
    print(yhat)
    print('*****************')
    F1scores = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(f1_score, average="macro"))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
    print('F1scores: ', F1scores)
    print(F1scores.mean())
    Accuracy = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(accuracy_score))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
    print('Accuracy: ', Accuracy)
    print(Accuracy.mean())
    Precision = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(precision_score, average="macro"))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
    print('Precision: ', Precision)
    print(Precision.mean())
    Recall = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(recall_score, average="macro"))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
    print('Recall: ', Recall)
    print(Recall.mean())


# In[2]:


def AdaBoostCode(df):
    print("""
print('¡¡¡WARNING!!!: You need imputate dataset before apply this function and check all data are numerical')
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
classname = input('Name of the class column:')
ts = float(input('write test size: '))
rs = int(input("write random_state(default 0, most common 4 or 2): "))
y = np.asarray(df[classname])
X = np.asarray(df.loc[:, df.columns != classname])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
print('Train size: ', X_train.shape, y_train.shape)
print('Test size: ', X_test.shape, y_test.shape)
print('********')
avg = input('Select averagestring : (‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]')
NE = int(input('Number of Estimators:'))
CV = int(input('cv or cross-validation generator or an iterable / optional / Most frequent 5 or 10:'))
yhat=[]
clf = AdaBoostClassifier(n_estimators=NE)
clf.fit(X, y)
yhat1 = clf.predict(X_test)
yhat.append(yhat1)
print('yhat:')
print(yhat)
print('*****************')
F1scores = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(f1_score, average="macro"))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
print('F1scores: ', F1scores)
print(F1scores.mean())
Accuracy = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(accuracy_score))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
print('Accuracy: ', Accuracy)
print(Accuracy.mean())
Precision = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(precision_score, average="macro"))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
print('Precision: ', Precision)
print(Precision.mean())
Recall = cross_val_score(clf, X, y, cv=CV, scoring = make_scorer(recall_score, average="macro"))   #cross_val_score: te da el score (clasificador, dataset X e y, cv =numero de k,scoring=)
print('Recall: ', Recall)
print(Recall.mean())
    """)


# In[ ]:




