
# coding: utf-8

# # Detección de Idiomas en el Abstract

# In[1]:


#Lectura de Librerías
import pandas as pd
import time
pd.set_option('display.max_columns', None)


# In[2]:


#Leyendo los abstracts
df = pd.read_csv(r'Data\abstract_eng.csv',sep=',')


# ## Remover Stopwords 

# In[3]:


import nltk
from nltk import *
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[4]:


stop = stopwords.words('english')


# In[24]:


#Se remueve el dato nan
df = df[df['Abstract_ing'].isna()==False]


# In[27]:


Abstracts = df[['Title','Abstract_ing']].drop_duplicates()
Abstracts.reset_index(inplace=True)
Abstracts["Abstract_ing"] = Abstracts["Abstract_ing"].str.lower().str.split()
#Abstracts.shape


# In[28]:


start = time.time()
Abstracts['Clean_Abstract'] = Abstracts[['Abstract_ing']] .apply(lambda x: [item for item in x if item not in stop])
end = time.time()
print((end-start)/60)


# In[29]:


Abstracts['Clean_Abstract'] = Abstracts['Abstract_ing'] .apply(lambda x: ' '.join([word for word in x]))


# In[30]:


#Abstracts[10:20]


# The common way of doing this is to transform the documents into tf-idf vectors, 
# then compute the cosine similarity between them. Any textbook on information retrieval (IR) covers this. See esp. Introduction to Information Retrieval, which is free and available online.
# 
# Tf-idf (and similar text transformations) are implemented in the Python packages
# Gensim and scikit-learn. In the latter package, computing cosine similarities is as easy as
# 

# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


start=time.time()
abstracts = df['Abstract_ing']
#stopwords_ = [word.decode('utf-8') for word in stopwords.words('english')]
tfidf = TfidfVectorizer().fit_transform(abstracts)
# no need to normalize, since Vectorizer will return normalized tf-idf
end = time.time()
print((end-start)/60)


# # La siguiente es la actividad que más recursos consume

# In[33]:


start=time.time()
pairwise_similarity = tfidf * tfidf.T
end = time.time()
print((end-start)/60)


# In[ ]:


type(pairwise_similarity)


# In[ ]:


#Para guardar la la matriz dispersa
#import scipy.sparse


# In[ ]:


#start=time.time()
#scipy.sparse.save_npz('Data/pairwise_similarity.npz', pairwise_similarity)
#end = time.time()
#print((end-start)/60)


# In[ ]:


#pairwise_similarity


# In[ ]:


#Load Matrix
#pairwise_similarity = scipy.sparse.load_npz('Data/pairwise_similarity.npz')


# In[ ]:


#pairwise_similarity.A


# ### Otras posibles Opciones
# https://stackoverflow.com/questions/51591510/text-similarity-approaches-do-not-reflect-real-similarity-between-texts
# https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
# https://stackoverflow.com/questions/101569/algorithm-to-detect-similar-documents-in-python-script

# #Apuntes
# #https://stackoverflow.com/questions/25443802/unicode-warning-when-using-nltk-stopwords-with-tfidfvectorizer-of-scikit-learn

# In[ ]:


import numpy as np
from sklearn.cluster import SpectralClustering


# In[ ]:


mat= np.matrix(pairwise_similarity.A)


# In[ ]:



temp_dict={}


# In[ ]:


start = time.time()
for i in range(10,20,1):
    if (i==13):
        continue
    nombre = "C_" + str(i)
    Abstracts_Category = SpectralClustering(i, affinity = 'precomputed').fit_predict(mat)
    temp_dict[nombre] = Abstracts_Category
    end = time.time()
    #print((end-start)/60)


# In[ ]:


df_category = pd.DataFrame(temp_dict)


# In[ ]:


#df_category


# In[ ]:


df_temp=pd.concat([Abstracts,df_category], axis=1)


# In[ ]:


#df_temp = df_temp[['Title','C_10','C_11','C_12','C_13','C_14','C_15','C_16','C_17','C_18','C_19','C_20']]
#df_final = df.merge(df_temp, how='right', on = 'Title')


# In[ ]:


#df_final.drop(columns='Unnamed: 0', axis=1, inplace=True)


# In[ ]:


#df_final


# In[ ]:


#df_final.to_csv(r'Data/scopus_cluster.csv')

