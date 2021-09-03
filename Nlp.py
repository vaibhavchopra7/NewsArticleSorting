#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd   #pandas library for data frame
datatrain=pd.read_csv(r"C:\Users\welcome\Desktop\BBC News Train.csv")  #traindata
datatest=pd.read_csv(r"C:\Users\welcome\Desktop\BBC News Test.csv")    #test data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
corpust = []
for i in range(0, 1490):                                           #making traindata text efficient
  text = re.sub('[^a-zA-Z]', ' ', datatrain['Text'][i])
  text = text.lower()
  text = text.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
  text = ' '.join(text)
  corpus.append(text)
for i in range(0, 735):                                           #making testdata text efficient
  textt = re.sub('[^a-zA-Z]', ' ', datatest['Text'][i])
  textt = textt.lower()
  textt = textt.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  textt = [ps.stem(word) for word in textt if not word in set(all_stopwords)]
  textt = ' '.join(textt)
  corpust.append(textt)
from sklearn.feature_extraction.text import CountVectorizer    #feature extraction
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()      #fit and transform features in train data
x = cv.transform(corpust).toarray()         # transform features of test data
y = datatrain.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder    # label for category
a=LabelEncoder()
Y=a.fit_transform(y)
from sklearn.linear_model import LogisticRegression    #logistic regression for classificaton with(97% accuracy)

lg=LogisticRegression(random_state=0).fit(X,Y)
e=lg.predict(x)
# print(a.inverse_transform(e))  #predicting 


# datatest["Pr"]=a.inverse_transform(e)  # prediction of test dataset goes to Finalt.csv file
# datatest.to_csv("Finalt.csv",index=False)   


# In[18]:


#for input
ap=[]
sentence = re.sub('[^a-zA-Z]', ' ', str(input()))
sentence = sentence.lower()
sentence = sentence.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
sentence = [ps.stem(word) for word in sentence if not word in set(all_stopwords)]
sentence = ' '.join(sentence)
ap.append(sentence)
st=cv.transform(ap).toarray()
print(a.inverse_transform(lg.predict(st)))

