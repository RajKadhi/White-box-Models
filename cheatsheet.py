
# Find the mean, count, median of salary with repective to the place

df.groupby("City").Salary.agg(["mean","median","count"])

### EDA
sns.heatmap(data.corr())
pandas_profiling.ProfileReport(data)

#
# matplotlib.inline 
# For Histogram
df.Salary.plot(kind="hist")
# Pie
df.Salary.plot(kind="pie")
# Crosstab show the frequency with which certain groups of data appear
pd.crosstab(data["job"],data["y"]).div(pd.crosstab(data["job"],data["y"]).sum(1), axis =0).plot(kind = "bar", stacked= True)
#
num_col = df.select_dtypes(include=np.number).columns
num_col = df.select_dtypes(exclude=np.number).columns


####################################################################################################
############################ sklearn.preprocessing Encoder
####################################################################################################
#One hot encoding
encoded_cat_col = pd.get_dummies(df[cat_col])
df_ready_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)

## Label Encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in cat_col:
    df[i] = label_encoder.fit_transform(df[i])

#Performng standard scaling ( zscore)
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler().fit_transform(df)

#Performng MinMax scaling
from sklearn.preprocessing import MinMaxScaler
minmax_scale = MinMaxScaler().fit_transform(df)
# Fit makes the mean to 0 by doing (x - mean)/stdv) now we do transform to use the same parameter of mean and stdv and apply it on the data.


#Read data from a csv
import pandas as pd
chipo = pd.read_csv(r"//path/chipotle.tsv",sep='\t')

# Sort the dataframe
data = data.sort_values(ascending = False)

#Turn the item price into a float
chipo["item_price"] = chipo["item_price"].apply(lambda x : x[1:]).astype('float')

dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)

# Multiple dataframe columns
chipo['revenue'] = chipo['quantity']* chipo['item_price']

df.rename(columns={'place':'city'},inplace = True)
df.replace({'tata consultancy':'tata',"Bank of america':'bofa'},inplace = True)

# Filter the data with age>40 and Salary<5000
df[(df.Age>40) & (df.Salary<5000)]

#Getting the max value of columns using for loop and lambda
for i in df.columns:
    print(df[i].max())

df.apply(lambda x : x.max(),axis=0)

# Changing the datatype of a columns
df["Salary"] = df["Salary"].astype('int')
df["Gender"] = df["Gender"].astype('int')

####################################################################################################
############################ Mean , Median , Mode value imputer
####################################################################################################

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
X = imp.fit_transform(X)

####################################################################################################
############################ Find Outliers 
####################################################################################################
#Script to find the outliers
for col_name in insurance.select_dtypes(include=np.number).columns[:-1]:
    print(col_name)
    q1 = insurance[col_name].quantile(0.25)
    q3 = insurance[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    print("Outliers = ",insurance.loc[(insurance[col_name] < low) | (insurance[col_name] > high), col_name])
    
 

#Script to exclude the outliers
for col_name in insurance.select_dtypes(include=np.number).columns[:-1]:
    print(col_name)
    q1 = insurance[col_name].quantile(0.25)
    q3 = insurance[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr 
    print("Exclude the Outliers = ",insurance.loc[~((insurance[col_name] < low) | (insurance[col_name] > high)), col_name])
    insurance[col_name] = insurance.loc[~((insurance[col_name] < low) | (insurance[col_name] > high)), col_name]
    
    
    
#Script to impute the outliers with median
for col_name in insurance.select_dtypes(include=np.number).columns[:-1]:
    print(col_name)
    q1 = insurance[col_name].quantile(0.25)
    q3 = insurance[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr 
    print("Change the outliers with median ",insurance[col_name].median())
    insurance.loc[(insurance[col_name] < low) | (insurance[col_name] > high), col_name] = insurance[col_name].median()
    
#Detecting outlier with z-score
outliers=[]
def detect_outlier(insurance):
    
    threshold=3
    mean_1 = np.mean(insurance)
    std_1 = np.std(insurance)
    
    
    for y in insurance:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = insurance.select_dtypes(include=np.number).apply(detect_outlier)
####################################################################################################
############################ Liner Regression Simple
####################################################################################################


X = df[["Age"]]
y = df["Salary"]

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 8 )


from sklearn.linear_model import LinearRegression
lin = LinearRegression()

lin.fit(train_X,train_y)

lin.intercept_
lin.coef_

train_pred = lin.predict(train_X)

from sklearn.metrics import mean_squared_error
print("Train MSE :",mean_squared_error(train_y,train_pred))
# Predict for the test value
test_pred = lin.predict(test_X)
print("Test MSE :",mean_squared_error(test_y,test_pred))

# to see what co eofficients are assigned to the columns in Liner regression use list zip method
list(zip(train_X.columns,lin.coef_))

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)
pol_reg.preditc(X_poly)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
####################################################################################################
############################ Logistic Regression 
####################################################################################################

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.3, random_state = 8)


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(train_X, train_y)
log_model.intercept_
log_model.coef_
train_pred = log_model.predict(train_X)
# To display the probability value of a prediction
log_model.predict_proba(train_X)

# Info: teh defaut treshold for predict_proba is 0.5. If we want to channge the threshold, write a wrapper class to validate 
# np.where predict.probs > user_def_threshold

# Error validation Logistic regression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
confusion_matrix(train_y,train_pred )
accuracy_score(train_y,train_pred)
recall_score(train_y,train_pred)
print(classification_report(train_y,train_pred))

## Plot ROC Curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_y, log_model.predict(test_X))
fpr, tpr, thresholds = roc_curve(test_y, log_model.predict_proba(test_X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

####################################################################################################
############################ Logistic Regression Ridge Lasso Elasticnet Regression
####################################################################################################
# These are called as Regularization techiniqe; L1 will make either 0 or 1 ; L2 (ridge will make closer to 0. Default is L2)
Lasso = LogisticRegression(penalty='l1')
Ridge = LogisticRegression(penalty='l2')

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss="log", penalty='elasticnet', max_iter=200000, learning_rate="optimal", eta0=0.001)
# eta0 is the initial learning rate for the constant

####################################################################################################
############################ KNN AND NAIVE BAYES 
####################################################################################################

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(train_X,train_y)
test_pred = knn.predict(test_X)


from sklearn.metrics import accuracy_score
accuracy_score(train_pred,train_y)


knn = KNeighborsClassifier(n_neighbors=57)
model_fit(knn, X, y)
model_fit(knn, X_scaled, y)
model_fit(knn, X_normalised, y)
##
KNeighborsClassifier(n_neighbors = K_value, weights='uniform', metric = 'manhattan', algorithm='auto',n_jobs = 4)

### Tuning KNN
# Neighbors tuning, Just vary the k value n_neighbors and predict the accuracy.
for k in range(1,12):
    print('Accuracy score on kNN using n_neighbours = {0}:'.format(2**k), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), 2**k)

# ### Metric tuning

for metric in ['euclidean', 'cosine', 'manhattan', 'chebyshev']:
    print('Accuracy score on kNN using {} metric and {} neighbours:'.format(metric,k), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), 2, metric)


# ### Weighted kNN

for weights in ['uniform', 'distance']:
    print('Accuracy score on kNN using weights = {0}:'.format(weights), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), 2, 'chebyshev', weights = weights)

#### Naive bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB, BernoulliNB
nb = GaussianNB()
model_fit(nb, X, y)
model_fit(nb, X_normalised, y)

####################################################################################################
############################ Under Sampling
####################################################################################################

from imblearn.under_sampling import NearMiss
pip install imblearn
undersample = NearMiss()
x_undersample, y_undersample = undersample.fit_sample()

####################################################################################################
############################ Hierarchial clustering - Dendogram
####################################################################################################
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(15,6))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.hlines(y=30,xmin=0,xmax=2000,lw=3,linestyles='--')
plt.text(x=900,y=20,s='Horizontal line crossing 7 vertical lines',fontsize=20)
#plt.grid(True)
dendrogram = sch.dendrogram(sch.linkage(df, method = 'centroid'))
plt.show()

####################################################################################################
############################ KMeans Example with Elbow
####################################################################################################

import random    # checkk
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs   # checkk
import pylab as pl      # checkk


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
kn.inertia_    #checkk  same as wcss - Withing cluster sum of squares


## Elbow Curve
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()
# We assign the labels to each row in dataframe.
df["Clus_km"] = labels
# We can easily check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()
# Now, lets look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

####################################################################################################
############################ Decision Tree 
####################################################################################################

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini',max_depth=10)
dt.fit( X_undersample, y_undersample)
X_undersample_predict = dt.predict(X_undersample)
accuracy_score(y_undersample, X_undersample_predict)
train_pred_prob, test_pred_prob = model_fit(dt, X_undersample, y_undersample, roc = True, conf = True, threshold=0.3)
## To plot the trees 
from dtreeplt import dtreeplt

type_criterion = ["gini", "entropy"]
def check_dept(type_criterion):
    train_acc_list= []
    test_acc_list = []
    for i in type_criterion:
        print("Max_ Depth ----------------------- = ",i)
        dt = DecisionTreeClassifier(criterion=i, max_depth=9)
        train_acc, test_acc = model_fit_dt(dt, X_undersample, y_undersample, roc = False, conf = False, threshold=0.3)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list

####################################################################################################
############################ Random Forest
####################################################################################################

from sklearn.ensemble import RandomForestClassifier
random_classifier = RandomForestClassifier(n_estimators=70)
random_classifier.fit(X_undersample, y_undersample)
X_undersample_pred_random = random_classifier.predict(X_undersample)
accuracy_score(y_undersample, X_undersample_pred_random)
random_classifier.feature_importances_
X_undersample.columns
feature_imp = pd.DataFrame([X_undersample.columns,random_classifier.feature_importances_]).T
feature_imp.columns = ["columns","importance"]
feature_imp
feature_imp.sort_values(by=['importance'],ascending=False, inplace=True)
feature_imp.head(15).plot(x = "columns", kind = "bar")
imp_col = feature_imp.head(10)["columns"]
model_fit(random_classifier,X_undersample[imp_col],y_undersample,roc=True,conf=True)
model_fit(random_classifier,X_undersample,y_undersample,roc= True)

####################################################################################################
############################ Adaboost 
####################################################################################################

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier()
ada.fit(X_undersample,y_undersample)
ada_pred = ada.predict(X_undersample)
accuracy_score(y_undersample, ada_pred)

model_fit(ada,X_undersample, y_undersample, roc=True, conf=True)
n_tree = [40,50,60,70,80,90,100]
for i in n_tree:
    ada_tune = AdaBoostClassifier(n_estimators=i)
    print("---------------Number of iteration------------ = ",i)
    model_fit(ada_tune, X_undersample,y_undersample,roc=True, conf=True)

ada.feature_importances_
feature_imp = pd.DataFrame([X_undersample.columns,ada.feature_importances_]).T
feature_imp.columns = ["columns","importance"]
feature_imp.sort_values(by=['importance'],ascending=False, inplace=True)
feature_imp.head(15).plot(x = "columns", kind = "bar")
imp_col = feature_imp.head(10)["columns"]
model_fit(ada,X_undersample[imp_col], y_undersample, roc=True, conf=True)
'max_features': [ 0.2, 0.3, 0.4,0.6,0.8,1  ], 'min_samples_split': [18,19,20],


####################################################################################################
############################ Grid Search and randamized search
####################################################################################################
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = [
{ 'max_features': [ 0.2, 0.3, 0.4,0.6,0.8,1  ], 'min_samples_split': [18,19,20],
   'max_depth' : [7,8,9]
}
]

tree = DecisionTreeClassifier()

grid_search = GridSearchCV(tree, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_undersample, y_undersample)

grid_search.best_estimator_
grid_search.best_score_

grid_search = RandomizedSearchCV(tree, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_undersample, y_undersample)
grid_search.best_estimator_

####################################################################################################
############################ Save and Load Model - Pickle and Joblib
####################################################################################################
# Assume "model" is the actual linear or logistic regression model. Then in order to save it use 

import pickle
file1 = 'open(model1save.pkl,wb)'  # Pickle files are binary stream so 'wb' meaning write binary
pickle.dump(model,file1)
file2 = 'opne(model1save.pkl,rb)'   # Pickle files are binary stream so 'rb' meaning read binary
pickle.load(model,file2)

import joblib    # joblib is more efficient when the model is complex and too many parameters
# since it uses numpy array efficeintly. We can think of Neural networks.
file1 = 'open(model1save.pkl,wb)'  # Pickle files are binary stream so 'wb' meaning write binary
joblib.dump(model,file1)
file2 = 'opne(model1save.pkl,rb)'   # Pickle files are binary stream so 'rb' meaning read binary
joblib.load(model,file2)

####################################################################################################
############################ Level 2 - NLP - Cheat sheet
####################################################################################################

a = "python is great"
print('ID:', id(a))      # shows the object identifier (address)
print('Type:', type(a))  # shows the object type
print('Value:', a)       # shows the object value
simple_string = 'Hello!' + " I'm a simple string"   # string concatenation
multi_line_string = """Hello I'm
a multi-line
string!"""
# Normal string with escape sequences leading to a wrong file path!
escaped_string = "C:\the_folder\new_dir\file.txt"                           # when the string has back slashes \ , it will be treated as escape char.
print(escaped_string)  # will cause errors if we try to open a file here
# raw string keeping the backslashes in its normal form
raw_string = r'C:\the_folder\new_dir\file.txt'
# Unincode
string_with_unicode = 'H\u00e8llo!'    # this will store value "Hello"
more_unicode = 'I love Pizza 🍕!  Shall we book a cab 🚕 to get pizza?'

####################################################################################################
############################ STRING Operation
####################################################################################################

s3 = ('This '
      'is another way '
      'to concatenate '
      'several strings!')
s3
# ### Substring value exist check
'way' in s3
'way' not in s3
len(s3)
s = 'PYTHON'

s[0], s[1], s[2], s[3], s[4], s[5]
s[-1], s[-2], s[-3], s[-4], s[-5], s[-6]
s[:]       # returns entire string
s[1:4]     # Retruns value starting from 1st byte to 4 byte. index starts from 0 ,1,2,3,4 
s[:3], s[3:]
s[-3:]      # It will return rest of the string starting last 3 digit "PYTHON" will return "HON"
s[:3] + s[3:]
s[:3] + s[-3:]
s[::1]  # no offset  
s[::2]  # print every 2nd character in string
s[::-1]  # reverses the string ""
for i, j in enumerate(a):
    print('Character ->', j, 'has index->', i)
# strings are immutable hence assignment throws error
s[0] = 'X'
print('Original String id:', id(s))
# creates a new string
s = 'X' + s[1:]
id(s)       # id address changes when you add data to string.
s = 'python is great'
s.capitalize()     # Python is great
s.title()          # Python Is Great
s.upper()           # PYTHON IS GREAT
s.lower()           # python is great
s.replace('python', 'NLP')      # NLP is great
'12345'.isdecimal()             # TRUE
'apollo11'.isdecimal()          # FALSE
'python'.isalpha()              # TRUE  # its is Alphanumeric
'number123'.isalpha()           # TRUE
# ## String splitting and joining
s = 'I,am,a,comma,separated,string'
s.split(',')                    # Will create a list ['I' , 'am' , 'a' , 'comma' , 'seperated', 'string']
' '.join(s.split(','))          # 'I am a comma separated string'
# stripping whitespace characters
s = '   I am surrounded by spaces    '
s.strip()               # 'I am surrounded by spaces'
sentences = 'Python is great. NLP is also good.'
sentences.split('.')
print('\n'.join(sentences.split('.')))   # concatenate the strings into seperate lines.
# ## Formatting strings using the format method - new style
'Hello {} {}, it is a great {} to meet you at {}'.format('Mr.', 'Jones', 'pleasure', 5)
'The {animal} has the following attributes: {attributes}'.format(animal='dog', attributes=['lazy', 'loyal'])
####################################################################################################
############################ Import re
####################################################################################################
s1 = 'Python is an excellent language'
s2 = 'I love the Python language. I also use Python to build applications at work!'
import re

pattern = 'python'
# match only returns a match if regex match is found at the beginning of the string
re.match(pattern, s1)
# pattern is in lower case hence ignore case flag helps
# in matching same pattern with different cases
re.match(pattern, s1, flags=re.IGNORECASE)
# printing matched string and its indices in the original string
m = re.match(pattern, s1, flags=re.IGNORECASE)
print('Found match {} ranging from index {} - {} in the string "{}"'.format(m.group(0), 
                                                                            m.start(), 
                                                                            m.end(), s1))
# only keeping words and removing special characters
words = list(filter(None, [re.sub(r'[^A-Za-z]', '', word) for word in words]))

# illustrating find and search methods using the re module
re.search(pattern, s2, re.IGNORECASE)
re.findall(pattern, s2, re.IGNORECASE)
match_objs = re.finditer(pattern, s2, re.IGNORECASE)
for m in match_objs:
    print('Found match "{}" ranging from index {} - {}'.format(m.group(0), 
                                                               m.start(), m.end()))
# illustrating pattern substitution using sub and subn methods
re.sub(pattern, 'Java', s2, flags=re.IGNORECASE)
# subn will give a tuple of the substituted text as well the counter of number of time the pattern is substituted.
re.subn(pattern, 'Java', s2, flags=re.IGNORECASE)
# dealing with unicode matching using regexes
s = u'H\u00e8llo! this is Python 🐍'
re.findall(r'\w+', s)
re.findall(r"[A-Z]\w+", s)
emoji_pattern = r"['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
re.findall(emoji_pattern, s, re.UNICODE)
####################################################################################################
############################ from collections import counter
####################################################################################################
# ### Finding the top ten most common words
from collections import Counter
words = [word.lower() for word in words]
c = Counter(words)
c.most_common(10)

####################################################################################################
############################ nltk.corpus.stopwords.words('english')
####################################################################################################
import nltk 

stopwords = nltk.corpus.stopwords.words('english')
stopwords[:10]
words = [word.lower() for word in words if word not in stopwords]
c = Counter(words)
c.most_common(10)
####################################################################################################
############################        textblob
####################################################################################################

from textblob import TextBlob
## creating a textblob object
blob = TextBlob("Inceptez is a great platform to learn data science.")
## textblobs are like python strings
blob[1:5]
blob.upper()
blob2 = TextBlob("It also helps community through blogs, hackathons, discussions,etc.")
## concat
blob + " And " + blob2
# ## Tokenization
blob = TextBlob("Inceptez is a great platform to learn data science.\n It helps community through blogs, hackathons, discussions,etc.")
blob.sentences
blob.sentences[0]    # gives first sentence
blob.sentences[0].words   # returns list of words in the first sentence
for words in blob.sentences[0].words:
    print (words)
# ## tags extraction .. It reutrns very word and its tags. JJ - Adjective , NN - Noun , NNS - Noun in plural NNP - Proper noun , NNPS , 
blob = TextBlob("Analytics Vidhya is a great platform to learn data science.")
for np in blob.tags:
    print (np)
for words, tag in blob.tags:
    print (words, tag)
# ## Sentiment Analysis
print (blob)
blob.sentiment  # will print Polarity ( float -1 to 1 neg -> pos) and subjectivity(float 0 to 1 emotions, opnions opposite to objective). 
# ## Word Inflection and Lemmatization
blob = TextBlob("Inceptez is a great platform to learn data science. \n It helps community through blogs, hackathons, discussions,etc.")
print (blob.sentences[1].words[1])
print (blob.sentences[1].words[1].singularize())  # "joke" is singularize and "jokes" is pluralize
from textblob import Word
w = Word('Platform')
w.pluralize()
## using tags
for word,pos in blob.tags:
    if pos == 'NN':
        print (word.pluralize())
## lemmatization
w = Word('jokes')
w.lemmatize()  ## v here represents verb
### Ngrams
blob = TextBlob("I went to sri lanka")
for ngram in blob.ngrams(2):
    print (ngram)
# ## Spelling correction
blob = TextBlob('Inceptez is a gret platfrm to lern data scence')
blob.correct()
blob.words[3].spellcheck()
import random
# ## Language Translation
blob = TextBlob('Hi Hope you are doing good?')
blob.translate(to ='es')       ## to spanish
blob1.translate(from_lang='ar', to ='en')
blob.translate(to= 'en')       ## to english
blob1 = TextBlob('هذا رائع')   ## arabic
blob1.detect_language()        ## Detects the language as arabic
blob = TextBlob("¿Hola como estás?")

####################################################################################################
############################      Text Classification using textblob
####################################################################################################
## in below list 'pos' , 'neg' are nothing but lables. It can even be 'a','b','c'
training = [
            ('Tom Holland is a terrible spiderman.','neg'),
            ('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','neg'),
            ('The Dark Knight Rises is the greatest superhero movie ever!','pos'),
            ('Fantastic Four should have never been made.','neg'),
            ('Wes Anderson is my favorite director!','pos'),
            ('Captain America 2 is pretty awesome.','pos'),
            ('Let\s pretend "Batman and Robin" never happened..','neg'),
            ]
testing = [
           ('Superman was never an interesting character.','neg'),
           ('Fantastic Mr Fox is an awesome film!','pos'),
           ('Dragonball Evolution is simply terrible!!','neg')
           ]

from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)
print (classifier.accuracy(testing))
classifier.show_informative_features(3)
blob = TextBlob('I like spiderman ', classifier=classifier)
print (blob.classify())
### Textblob classifier classifies very poorly. Ex "Captain America is very bad " will be 'pos' because two words matches.

####################################################################################################
############################      cosine similarity
####################################################################################################
from sklearn.metrics.pairwise import cosine_similarity

####################################################################################################
############################      Deep Learning Solution
####################################################################################################

