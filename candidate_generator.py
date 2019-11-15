#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split


# This is an example from https://www.freecodecamp.org/news/how-to-extract-keywords-from-text-with-tf-idf-and-pythons-scikit-learn-b2a0f3d7e667/

# In[9]:


df_idf = pd.read_csv('amazon/reviews.csv')


# In[10]:


print("Schema:\n", df_idf.dtypes)
print("Shape of database =", df_idf.shape)


# In[11]:


def pre_process(text):
    # to lowercase
    text=text.lower()
    
    # remove tags
    text = re.sub("&lt;/?.*?&gt;", "&lt;&gt; ", text)
    
    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)
    
    return text


# In[12]:


df_idf['text'] = df_idf['title'] + " " + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x: pre_process(str(x)))


# In[13]:


df_idf['text'][2]


# In[14]:


def get_stop_words(stop_file_path):
    with open(stop_file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


# In[15]:


stopwords = get_stop_words('stopwords.txt')
docs = df_idf['text'].tolist()


# In[16]:


cv = CountVectorizer(max_df = .85, stop_words=stopwords)
wordCountVec = cv.fit_transform(docs)


# In[17]:


list(cv.vocabulary_.keys())[:10]


# In[18]:


tfidf = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf.fit(wordCountVec)


# In[19]:


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1] , x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    
    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
        
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
            
        return results


# In[20]:


feature_names = cv.get_feature_names()

doc = docs[1]

tf_idf_vector = tfidf.transform(cv.transform([doc]))

sorted_items = sort_coo(tf_idf_vector.tocoo())

keywords = extract_topn_from_vector(feature_names, sorted_items, 10)


# In[21]:


for idx in range(len(sorted_items)):
    print(feature_names[sorted_items[idx][0]], sorted_items[idx][1])


# In[22]:


y = df_idf['rating']
# fixing the labels, if > 3.5 is going to be 1 which is positive, else 0
y = y.apply(lambda x: 1 if x > 3.5 else 0) 
y = y.to_numpy()
x = wordCountVec.toarray()
print(x.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[24]:


print(type(X_train), type(y_train))


# In[ ]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)


# In[ ]:




