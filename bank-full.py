import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

simplefilter(action='ignore', category=FutureWarning)

url ='bank-full.csv'
data = pd.read_csv(url)
data.drop(['balance', 'duration', 'pdays'],axis=1, inplace=True)

data['job'].replace(['blue-collar','management','technician','admin.',
'services','retired','self-employed','entrepreneur','unemployed',
'housemaid','student','unknown'], [0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)

data['marital'].replace(['married','single','divorced'], [0, 1, 2], inplace=True)

data['education'].replace(['secondary','tertiary', 'primary','unknown'], [0, 1, 2, 3], inplace=True)

data.default.replace(['no', 'yes'], [0, 1], inplace=True)

data.housing.replace(['yes', 'no'], [0, 1], inplace=True)

data.loan.replace(['no', 'yes'], [0, 1], inplace=True)

data['contact'].replace(['cellular','unknown','telephone'], [0, 1, 2], inplace=True)

data['month'].replace(['may','jul','aug','jun','nov','apr','feb','jan',
                     'oct','sep','mar','dec'], [0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)


data['poutcome'].replace(['unknown','failure', 'other','success'], [0, 1, 2, 3], inplace=True)

data.y.replace(['no', 'yes'], [0, 1], inplace=True)

data.age.replace(np.nan,41, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60,100]
nombres = ['1','2','3','4','5','6','7']
data.age=pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0,how='any',inplace=True)

#partir la tabla en dos
data_train = data[:22701]
data_test = data[22701:]

x=np.array(data_train.drop(['y'], 1))
y=np.array(data_train.y)# 0 sale 1 no sale

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)# 0 sale 1 no sale