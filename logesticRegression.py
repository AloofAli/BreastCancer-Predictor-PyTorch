import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import torch
import torch.nn as nn

bc= datasets.load_breast_cancer()

X,y =bc.data , bc.target

n_samples, n_features= X.shape

X_train , X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))


y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

class LogisticRegression(nn.Module):
  def __init__(self,n_input_features):
    super(LogisticRegression,self).__init__()
    self.linear=nn.Linear(n_input_features,1)

  def forward(self,x):
    y_predicted=torch.sigmoid(self.linear(x))
    return y_predicted


model=LogisticRegression(n_features)



lr=0.01
criterion= nn.BCELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=lr)

num_epochs=100

for epoch in range(num_epochs):
  ypred=model(X_train)
  loss=criterion(ypred,y_train)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  if (epoch+1) %10 ==0:
    print(f"epoch= {epoch+1}, loss = {loss.item()}")





with torch.no_grad():
  y_prediction= model(X_test)
  y_prediction_cls=y_prediction.round()
  acc = y_prediction_cls.eq(y_test).sum() / float(y_test.shape[0])
  print(f" accuracy=  {acc}")
