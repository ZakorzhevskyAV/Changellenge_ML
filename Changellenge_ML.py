import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA

# подгружаем данные
data = read_csv('TestTest.csv')
cleandata = read_csv('contest_test.csv')

# удаляем столбцы, заполненные менее чем на 70 процентов
NameCol = []
i = 0
Quantil = data.shape[0]
while i <= 259:
    a = data.ID[data.iloc[:,i+2].notnull()].count()
    if a <= 70*Quantil/100:
        NameCol.append(i)
    i+=1
for i in NameCol:
        DelName='FEATURE_'+str(i)
        del data[DelName]

# проверка на корреляцию, удаляем сильно коррелирующие столбцы
CorrKoef = data.corr()
FielDrop = [i for i in CorrKoef if CorrKoef[i].isnull().drop_duplicates().values[0]]
CorrField = []
CorrDropFiels = []

# выискиваем все столбцы, которые коррелируют друг с другом >90%, cкладываем их в контейнер CorrDropFiel
for i in CorrKoef:
    for j in CorrKoef.index[CorrKoef[i] > 0.999]:
        if i != j and j not in CorrField and i not in CorrField:
            CorrField.append(j)
            CorrDropFiels.append(i), CorrDropFiels.append(j)
FielDrop = FielDrop + CorrDropFiels # объединяем контейнеры CorrDropFiels и FielDrop
newdata = data.drop(FielDrop, axis=1)# удаляем сразу все
newdata = newdata.fillna(data.median(axis=0),axis=0)
newdata.describe()
data_numerical = newdata.drop('ID',axis=1)
data_numerical = data_numerical.drop('TARGET',axis=1)
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
coder = PCA(n_components=65)
X = data_numerical
y = newdata.TARGET.values
feature_names = X.columns
X = coder.fit_transform(data_numerical)

# подготовка тестовой выборки
NameCol = []
i = 0
Quantil = cleandata.shape[0]
while i <= 259:
    a = cleandata.ID[cleandata.iloc[:,i+1].notnull()].count()
    if a <= 30*Quantil/100:
        NameCol.append(i)
    i+=1
for i in NameCol:
        DelName='FEATURE_'+str(i)
        del cleandata[DelName]
        #print(DelName)
cleandata = cleandata.drop(FielDrop, axis=1)
cleandata=cleandata.fillna(cleandata.median(axis=0),axis=0)
cleandata = cleandata.drop('ID',axis=1)
Xdata=coder.fit_transform(cleandata)
ydata=[]

#Разделим на обучающую и тестовую выборку
from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 11)

#Собственно ML:
from sklearn import ensemble
gbt = ensemble.GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, subsample=0.5,random_state=11)

#Обучение и тест тестовой модели
gbt.fit(X_train, y_train)
err_train = np.mean(y_train != gbt.predict(X_train))
predicted = gbt.predict(X_test)
expected = y_test
err_test = np.mean(y_test != gbt.predict(X_test))
print('Ошибка на обучающей выборке:',err_train, 'Ошибка на тестовой выборке:', err_test)
print(metrics.classification_report(expected, predicted))
print('Матрица смещений:')
print(metrics.confusion_matrix(expected, predicted))
print('Оценка точности в метрике macro F1_score:', metrics.f1_score(expected, predicted, labels=None, average='macro', sample_weight=None))
#Тест на выборке contest_test.csv
ydata = gbt.predict(Xdata)
print('Массив предсказанных номеров сегментов: ', ydata)
model.fit(X_train, y_train)
answerdata = model.predict(Xdata)
print(answerdata)