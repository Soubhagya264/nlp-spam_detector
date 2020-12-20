# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 22:48:46 2020

@author: HP
"""

import pandas as pd
message=pd.read_csv("smsspamcollection/SMSSpamCollection",sep='\t',names=["label","message"])
print(message.head)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps=PorterStemmer()

corpus=[]
for i in range(len(message)):
    review=re.sub('[^a-zA-Z]', ' ',message["message"][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(message['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)    

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
conf_Mat=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)



