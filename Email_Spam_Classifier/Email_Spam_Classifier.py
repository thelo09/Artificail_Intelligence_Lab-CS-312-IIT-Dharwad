import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
data = pd.read_csv('spambase.data', header=None)
X = data.values[:,:-1]
y = data.values[:, -1:]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_train = y_train.flatten()
y_test = y_test.flatten()
cvals_0 = [100000,90000,95000]
k=0
best=0
print("--"*40)
print("RBF:")
for x in cvals_0:
  print(f'C value is {x}')
  model = svm.SVC(kernel='rbf',C=x)
  model.fit(x_train, y_train)
  pred_train = model.predict(x_train)
  print(f'Training Accuracy = {model.score(x_train,y_train)}')
  print(f'Test Accuracy ={model.score(x_test,y_test)} ')
  if model.score(x_test,y_test) > k:
    k = model.score(x_test,y_test)
    best = x
print(f'Best accuracy is {k} and it is for C={best}')
p=0
print("--"*40)
print("Quadratic:")
cvals_1 = [1000]
for y in cvals_1:
  print(f'C value is {y}')
  model = svm.SVC(kernel='poly', degree=2 ,C=y)
  model.fit(x_train, y_train)
  pred_train = model.predict(x_train)
  print(f'Training Accuracy = {model.score(x_train,y_train)}')
  print(f'Test Accuracy ={model.score(x_test,y_test)} ')
  if model.score(x_test,y_test) > p:
    p = model.score(x_test,y_test)
    best = y
print(f'Best accuracy is {p} and it is for C={best}')
q=0
print("--"*40)
print("Linear:")
cvals_2 = [0.38,0.35,1]
for z in cvals_2:
  print(f'C value is {z}')
  model = svm.SVC(kernel='linear',C=z)
  model.fit(x_train, y_train)
  pred_train = model.predict(x_train)
  print(f'Training Accuracy = {model.score(x_train,y_train)}')
  print(f'Test Accuracy ={model.score(x_test,y_test)} ')
  if model.score(x_test,y_test) > q:
    q = model.score(x_test,y_test)
    best = z
print(f'Best accuracy is {q} and it is for C={best}')


