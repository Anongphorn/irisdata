import pickle #we use this lib for dumping the model to binary format

import pandas as pd # this is for create dataframe for easy manipulation
from sklearn.ensemble import RandomForestClassifier #i selected this as my meta-classfifer
from sklearn.metrics import accuracy_score #just for evaluation sake
from sklearn.model_selection import train_test_split# for simplest holdout sampling meothd

df=pd.read_csv(r"D:\Data science examination\everthing with data\01-Oct\iris.data")#we open the dataset and parse as dataframe

x=df.iloc[:,:-1]#here we select all row, and until the column before the last one,because the last on is
y=df.iloc[:,-1]#we select all row

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
classifer=RandomForestClassifier()
classifer.fit(X_train,y_train)

y_pred=classifer.predict(X_test)

score=accuracy_score(y_test,y_pred)
print(score)
pickle_out=open("model_iris.pkl","wb")
pickle.dump(classifer,pickle_out)
pickle_out.close()


