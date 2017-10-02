
# coding: utf-8

# # Web pages classification
# ## A machine learning and data mining task
# * Data: http://www.cs.cmu.edu/afs/cs/project/theo-20/www/data/
# 
# ### UFSC - DAS - MACHINE LEARNING - JOMI FRED HUBNER
# * Authors: Luis Felipe Pelison, Alex Cani and Iago Oliveira
#  
#  ** Helped by: https://github.com/justmarkham/pycon-2016-tutorial/blob/master/tutorial_with_output.ipynb

# ## 1. Importing data and python libraries

# In[1]:



import pandas as pd
import os
import os.path
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Para que os graficos apareçam
get_ipython().magic(u'matplotlib inline')


# ### Python script to put data on a csv file

# In[2]:


urls = []

for dirpath, dirnames, filenames in os.walk("./webkb"):
    for filename in filenames:
        urls.append(os.path.join(dirpath, filename))

print len(urls)

regex = r"\<([\s\S]*)>"

with open('corpus.csv', 'w') as output:
    for j, i in enumerate(urls):
        with open(i) as input:
            if j == 0:
                output.write("url-||-university-||-html-||-target \n")
            search = re.search(regex, input.read())
            if search: 
                html = search.group(0).replace('\n', ' ').replace('\r', '')
            else:
                html = 'NaN'
            target = i.split("/")[2]
            university = i.split("/")[3]
            url = i.split("/")[4]
            print j
            output.write("{0}-||-{1}-||-{2}-||-{3} \n".format(url, university, html, target))


# ### CSV to PANDAS

# In[3]:


#one = pd.read_html
path = './corpus.csv'
one = pd.read_csv(path, sep=r"\-\|\|-", quotechar='"', engine='python')
get_ipython().magic(u'time')


# In[4]:


one.head()


# ### Looking the data

# In[5]:


one.describe()


# ## Data Website :
# http://www.cs.cmu.edu/afs/cs/project/theo-20/www/data/
# 
# ### Number of data by universities
# 
# * Cornell - 867
# * Misc - 4120
# * Texas - 827
# * Washington - 1205
# * Wisconsin - 1263
# 
# ** sum ** = 8282
# 

# In[6]:


one.groupby(one['university']).count()


# ### Number of data by target class
# 
# * course - (930)
# * department - (182)
# * faculty - (1124)
# * other - (3764)
# * project - (504)
# * staff - (137)
# * student - (1641)
# 
# ** sum ** = 8282

# In[7]:


one.groupby(one['target']).count()


# ### Some html are 'NaN'.
# 
# * Let's delete them.
# 

# In[8]:


# Who is 'Nan'?

one[one['html'].isnull()]


# In[9]:


# Deleting rows with null values. There are 17 rows.

one = one.dropna(axis=0, how='any')
#one[one['html'].isnull()]
one.describe()


# ## 2.  Data Visualization

# #### Number of data by universities

# In[10]:


one.groupby(['university']).count().plot.bar();


# #### Number of data by target class

# In[11]:


one.groupby(['target']).count().plot.bar();


# In[12]:


one.groupby(['university','target'])['url'].count()


# #### Data by university AND target class

# In[13]:


temp = pd.crosstab(one['university'], one['target'])
temp.plot(kind='bar', stacked=False, grid=True, figsize=(15,7));


# ## 3. Machine Learning Models

# #### Remembering the data

# In[14]:


one.head()


# In[15]:


one.groupby(['target']).count()


# #### Create a new column with numeric values to the target

# In[16]:


one['target_num'] = one.target.map({'course':0, 'department':1, 'faculty':2, 'other':3, 'project':4, 'staff':5, 'student':6})


# #### HTML data

# In[17]:


# Define X and y (from the data) for use with COUNTVECTORIZER

X = one.html
y = one.target_num
print(X.shape)
print(y.shape)


# #### Split into training and testing sets

# In[111]:


# split X and y into training and testing sets

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Vectorizing

# In[112]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# instantiate the vectorizer

vect = CountVectorizer(encoding='latin-1', stop_words='english', ngram_range=(1, 2), min_df=5)
#vect = TfidfVectorizer(encoding='latin-1', stop_words='english', ngram_range=(1, 2), min_df=5)
# Esse tfIdf deu ruim

                       
# combine fit and transform into a single step

X_train_dtm = vect.fit_transform(X_train)
X_train_dtm


# In[113]:


# transform testing data (using fitted vocabulary) into a document-term matrix

X_test_dtm = vect.transform(X_test)
X_test_dtm


# ### Comparing models

# #### Naive Bayes

# In[114]:


# import and instantiate a Multinomial Naive Bayes model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[115]:


# train the model using X_train_dtm (timing it with an IPython "magic command")

get_ipython().magic(u'time nb.fit(X_train_dtm, y_train)')


# In[116]:


# make class predictions for X_test_dtm

y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# #### Logistic Regression

# In[117]:


# import and instantiate a logistic regression model

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')

# train the model using X_train_dtm

get_ipython().magic(u'time logreg.fit(X_train_dtm, y_train)')


# In[118]:


# make class predictions for X_test_dtm

y_pred_class = logreg.predict(X_test_dtm)


# calculate accuracy

metrics.accuracy_score(y_test, y_pred_class)


# In[119]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[120]:


one.groupby(['target_num', 'target']).count()


# #### Random Forest

# In[121]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=300)

get_ipython().magic(u'time forest.fit(X_train_dtm, y_train)')


# In[122]:



y_pred_class = forest.predict(X_test_dtm)

metrics.accuracy_score(y_test, y_pred_class)


# #### Gradient Boosting - NOT RUN!

# In[32]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)

get_ipython().magic(u'time clf.fit(X_train_dtm, y_train)')


# In[33]:


get_ipython().magic(u'time X_test_array = X_test_dtm.toarray()')


# In[34]:


# NAO RODAR! LERDO
y_pred_class = clf.predict(X_test_array)

metrics.accuracy_score(y_test, y_pred_class)


# ## 4. Examining tokenization

# In[123]:


# store the vocabulary of X_train

X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# In[124]:


# examine the first 50 tokens
print(X_train_tokens[0:50])


# In[125]:


# examine the last 50 tokens
print X_train_tokens[-50:]


# In[126]:


# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_


# In[127]:


nb.feature_count_.shape


# In[128]:


# number of times each token appears across all course
course_token_count = nb.feature_count_[0, :]
course_token_count

# number of times each token appears across all department
department_token_count = nb.feature_count_[1, :]
department_token_count

# number of times each token appears across all faculty
faculty_token_count = nb.feature_count_[2, :]
faculty_token_count

# number of times each token appears across all other
other_token_count = nb.feature_count_[3, :]
other_token_count

# number of times each token appears across all project
project_token_count = nb.feature_count_[4, :]
project_token_count

# number of times each token appears across all staff
staff_token_count = nb.feature_count_[5, :]
staff_token_count

# number of times each token appears across all student
student_token_count = nb.feature_count_[6, :]
student_token_count


# In[129]:


# create a DataFrame of tokens with their separate target counts
tokens = pd.DataFrame({'token':X_train_tokens, 'course':course_token_count, 'department':department_token_count, 'faculty':faculty_token_count, 'other':other_token_count, 'project':project_token_count, 'staff':staff_token_count, 'student':student_token_count}).set_index('token')
tokens[:50]


# In[130]:


# examine 5 random DataFrame rows
tokens.sample(8, random_state=10)


# # Training with an university (MISC) and testing on WISCONSIN university.

# In[131]:


misc = one[one.university == 'misc']
wisconsin = one[one.university == 'wisconsin']


# In[132]:


misc.shape, wisconsin.shape


# In[133]:


# Define X and y (from the data) for use with COUNTVECTORIZER

#Training data

X_misc = misc.html
y_misc = misc.target_num
X_misc.shape, y_misc.shape

#Testing data

X_wisconsin = wisconsin.html
y_wisconsin = wisconsin.target_num
X_wisconsin.shape, y_wisconsin.shape


# In[134]:


from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer

# instantiate the vectorizer

vect = CountVectorizer(encoding='latin-1', stop_words='english', ngram_range=(1, 2), min_df=5)
#vect = TfidfVectorizer(encoding='latin-1', stop_words='english', ngram_range=(1, 2), min_df=5)
# Esse tfIdf deu ruim

                       
# combine fit and transform into a single step

X_misc_dtm = vect.fit_transform(X_misc)
X_misc_dtm


# In[135]:


# transform testing data (using fitted vocabulary) into a document-term matrix

X_wisconsin_dtm = vect.transform(X_wisconsin)
X_wisconsin_dtm


# ## Machine Learning

# ### Random Forest

# In[138]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=300)

get_ipython().magic(u'time forest.fit(X_misc_dtm, y_misc)')

y_pred_wisconsin = forest.predict(X_wisconsin_dtm)

metrics.accuracy_score(y_wisconsin, y_pred_wisconsin)

