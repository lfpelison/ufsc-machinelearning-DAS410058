
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


# In[9]:


get_ipython().system(u'pwd')


# In[46]:


get_ipython().system(u'ls')


# ### Python script to put data on a csv file

# In[11]:


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

# In[13]:


#one = pd.read_html
path = './corpus.csv'
one = pd.read_csv(path, sep=r"\-\|\|-", quotechar='"', engine='python')
get_ipython().magic(u'time')


# In[14]:


one.head()


# ### Looking the data

# In[15]:


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

# In[16]:


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

# In[17]:


one.groupby(one['target']).count()


# ### Some html are 'NaN'.
# 
# * Let's delete them.
# 

# In[18]:


# Who is 'Nan'?

one[one['html'].isnull()]


# In[19]:


# Deleting rows with null values. There are 17 rows.

one = one.dropna(axis=0, how='any')
#one[one['html'].isnull()]
one.describe()


# ## 2.  Data Visualization

# #### Number of data by universities

# In[20]:


one.groupby(['university']).count().plot.bar();


# #### Number of data by target class

# In[21]:


one.groupby(['target']).count().plot.bar();


# In[26]:


one.groupby(['university']).count().plot.bar();


# #### Data by university AND target class

# In[27]:


temp = pd.crosstab(one['university'], one['target'])
temp.plot(kind='bar', stacked=False, grid=True, figsize=(15,7));


# #### Insights from the images above:
# 
# * <input type="checkbox"> Train in some universities and test in another. Ps. Department and Staff couldn't be a good ideia
# * <input type="checkbox"> Train in all universities and test in misc / The opposite too
# * <input type="checkbox" checked> Train random and test random 
# * <input type="checkbox"> Train all above with MISC and without MISC
# 
# * <input type="checkbox" checked> Train with URL, not HTML -> ** é pior **
# * <input type="checkbox"> Train with URL + HTML
#     * <input type="checkbox"> **Create new column with URL splited**
# 
# #### Other options
# 
# * <input type="checkbox"> Add my own stop_words (from sklearn.feature_extraction import text | stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words) )
# * <input type="checkbox" checked> TdIDF vectorizer -> ** é pior **
# * <input type="checkbox"> Feature selection
# * <input type="checkbox"> Search about cross validation
# * <input type="checkbox"> Weight to html tags?
# 

# ## 3. Machine Learning Models

# #### Remembering the data

# In[31]:


one.head()


# In[29]:


one.groupby(['target']).count()


# #### Create a new column with numeric values to the target

# In[30]:


one['target_num'] = one.target.map({'course':0, 'department':1, 'faculty':2, 'other':3, 'project':4, 'staff':5, 'student':6})


# #### HTML data

# In[32]:


# Define X and y (from the data) for use with COUNTVECTORIZER

X = one.html
y = one.target_num
print(X.shape)
print(y.shape)


# #### Split into training and testing sets

# In[33]:


# split X and y into training and testing sets

from sklearn.cross_validation import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Vectorizing

# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# instantiate the vectorizer

vect = CountVectorizer(encoding='latin-1', stop_words='english', ngram_range=(1, 2), min_df=5)
#vect = TfidfVectorizer(encoding='latin-1', stop_words='english', ngram_range=(1, 2), min_df=5)
# Esse tfIdf deu ruim

                       
# combine fit and transform into a single step

X_train_dtm = vect.fit_transform(X_train)
X_train_dtm


# In[35]:


# transform testing data (using fitted vocabulary) into a document-term matrix

X_test_dtm = vect.transform(X_test)
X_test_dtm


# ### Comparing models

# #### Naive Bayes

# In[36]:


# import and instantiate a Multinomial Naive Bayes model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[37]:


# train the model using X_train_dtm (timing it with an IPython "magic command")

get_ipython().magic(u'time nb.fit(X_train_dtm, y_train)')


# In[38]:


# make class predictions for X_test_dtm

y_pred_class = nb.predict(X_test_dtm)


# calculate accuracy of class predictions

from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# # --------------------------------------- #

# In[47]:


urls_test = []

for dirpath, dirnames, filenames in os.walk("./test"):
    for filename in filenames:
        urls_test.append(filename)

print len(urls_test)
print urls_test[:10]


# In[100]:


urls_test = []

path = './test_filenames.csv'

with open(path) as input:
    for j, i in enumerate(input.readlines()):
        if j == 0:
            pass
        else:
            urls_test.append(i.split(';')[1].replace('\n', ''))
            
print len(urls_test)
print urls_test[:10]
    


# In[104]:


regex = r"\<([\s\S]*)>"

with open('corpus_test.csv', 'w') as output:
    for j, i in enumerate(urls_test):
        path = os.path.join('./test/', i)
        with open(path) as input:
            if j == 0:
                output.write("url-||-html \n")
            search = re.search(regex, input.read())
            if search: 
                html = search.group(0).replace('\n', ' ').replace('\r', '')
            else:
                html = 'Não encontrado'
            url = i
            print j
            output.write("{0}-||-{1} \n".format(url, html))


# In[105]:


path = './corpus_test.csv'
corpus_test = pd.read_csv(path, sep=r"\-\|\|-", quotechar='"', engine='python')
get_ipython().magic(u'time')


# In[115]:


corpus_test.head(20)


# In[107]:


X_test_test = corpus_test.html
X_test_test = X_test_test.dropna(axis=0, how='any')
X_test_test


# In[108]:


X_test_test_dtm = vect.transform(X_test_test)
X_test_test_dtm


# In[109]:


# make class predictions for X_test_dtm

y_pred_test_class = nb.predict(X_test_test_dtm)
y_pred_test_class


# In[110]:


path = './test_classes.csv'
test_classes = pd.read_csv(path, sep=";", engine='python')
get_ipython().magic(u'time')

test_classes.head()


# In[120]:


test_classes['target_num'] = test_classes['class'].map({'course':0, 'departament':1, 'faculty':2, 'other':3, 'project':4, 'staff':5, 'student':6})
test_classes.head(50)


# In[119]:


corpus_test.head(50)


# In[112]:


y_test_test = test_classes.target_num


# In[113]:


# calculate accuracy of class predictions

from sklearn import metrics
metrics.accuracy_score(y_test_test, y_pred_test_class)


# #### Logistic Regression

# In[39]:


# import and instantiate a logistic regression model

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight='balanced')

# train the model using X_train_dtm

get_ipython().magic(u'time logreg.fit(X_train_dtm, y_train)')


# In[40]:


# make class predictions for X_test_dtm

y_pred_class = logreg.predict(X_test_dtm)


# calculate accuracy

metrics.accuracy_score(y_test, y_pred_class)


# In[41]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[42]:


one.groupby(['target_num', 'target']).count()


# In[121]:


# make class predictions for X_test_dtm

y_pred_test_class = logreg.predict(X_test_test_dtm)


# calculate accuracy

metrics.accuracy_score(y_test_test, y_pred_test_class)


# #### Random Forest

# In[43]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=300)

get_ipython().magic(u'time forest.fit(X_train_dtm, y_train)')


# In[44]:


y_pred_class = forest.predict(X_test_dtm)

metrics.accuracy_score(y_test, y_pred_class)


# In[122]:


# make class predictions for X_test_dtm

y_pred_test_class = forest.predict(X_test_test_dtm)


# calculate accuracy

metrics.accuracy_score(y_test_test, y_pred_test_class)


# #### Gradient Boosting - NOT RUN!

# In[125]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0)

get_ipython().magic(u'time clf.fit(X_train_dtm, y_train)')


# In[258]:


# NAO RODAR! LERDO
y_pred_class = clf.predict(X_test_dtm.toarray())

metrics.accuracy_score(y_test, y_pred_class)


# In[123]:


X_tt = X_test_test_dtm.toarray()


# In[127]:


y_pred_test_class = clf.predict(X_tt)

metrics.accuracy_score(y_test_test, y_pred_test_class)


# ## 4. Examining tokenization

# In[340]:


# store the vocabulary of X_train

X_train_tokens = vect.get_feature_names()
len(X_train_tokens)


# In[341]:


# examine the first 50 tokens
print(X_train_tokens[0:50])


# In[342]:


# examine the last 50 tokens
print X_train_tokens[-50:]


# In[343]:


# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_


# In[344]:


nb.feature_count_.shape


# In[345]:


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


# In[346]:


# create a DataFrame of tokens with their separate target counts
tokens = pd.DataFrame({'token':X_train_tokens, 'course':course_token_count, 'department':department_token_count, 'faculty':faculty_token_count, 'other':other_token_count, 'project':project_token_count, 'staff':staff_token_count, 'student':student_token_count}).set_index('token')
tokens[:50]


# In[347]:


# examine 5 random DataFrame rows
tokens.sample(8, random_state=10)

