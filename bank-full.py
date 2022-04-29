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