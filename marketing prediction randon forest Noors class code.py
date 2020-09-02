#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Necessary Library
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
#import pandas_profiling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report

import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[9]:


#load the data to dataframe
data = pd.read_csv("bank-additional-full.csv", sep = ";")


# # Preprocessing

# In[95]:


pandas_profiling.ProfileReport(data)


# In[10]:


data.duplicated().sum()


# In[11]:


data.drop_duplicates(inplace = True)


# In[12]:


data.isna().sum()


# In[13]:


data['y'].value_counts()


# In[14]:


data["y"].unique()


# In[15]:


data["y"].replace({"yes":1,"no":0}, inplace = True)


# ##### duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

# In[16]:


data.drop(columns="duration", inplace = True)


# In[17]:


data.columns


# In[18]:


data["month"].unique()


# In[19]:


data["day_of_week"].unique()


# In[20]:


data["job"].unique()


# In[21]:


data["job"].value_counts()


# In[22]:


data = data[data["job"]!="unknown"]


# In[23]:


data["job"].value_counts()


# In[24]:


data["marital"].unique()


# In[25]:


data["marital"].value_counts()


# In[26]:


data = data[data["marital"]!="unknown"]


# In[27]:


data["marital"].value_counts()


# In[28]:


data["education"].unique()


# In[29]:


data["education"].value_counts()


# In[30]:


np.where(data["education"]=="basic.9y","basic",data["education"])


# In[31]:


data["education"] = data["education"].replace({"basic.4y":"basic","basic.6y":"basic","basic.9y":"basic"})


# In[32]:


data["education"].value_counts()


# In[33]:


data.loc[data["education"]=="unknown","education"] = np.NAN


# In[34]:


data["education"].value_counts()


# In[35]:


data["education"].isna().sum()


# In[36]:


data["education"] = data.groupby("job")["education"].transform(lambda x : x.fillna(x.mode()[0]))


# In[37]:


data["default"].unique()


# In[38]:


data["default"].value_counts()


# In[39]:


pd.crosstab(data["default"],data["y"])


# In[40]:


data[data["default"]=="yes"]


# ##### Removing Default column as the labes are skewed towards "no" label 

# In[41]:


data.drop(columns = "default", inplace = True)


# In[42]:


data["housing"].unique()


# In[43]:


data["housing"].value_counts()


# In[44]:


data.loc[data["housing"]=="unknown","housing"] = np.NAN


# In[45]:


data.dropna(subset=["housing"], inplace= True)


# In[46]:


data["loan"].unique()


# In[47]:


data["contact"].unique()


# In[48]:


data["month"].unique()


# In[49]:


data["day_of_week"].unique()


# In[50]:


data["campaign"].unique()


# #### 999 - actually never called
# #### by real time data we have it denotes we called 999 days back it is less significant. So, Keeping as it is.
# 
# #### Note: Feel free to change it and use it

# In[51]:


data["pdays"].unique()


# In[52]:


data["previous"].unique()


# In[53]:


data["previous"].value_counts()


# In[54]:


pd.crosstab(data["previous"],data["y"])


# In[55]:


data["poutcome"].unique()


# # Explorartory Data Analysis(EDA)

# In[56]:


data.head()


# In[57]:


data.shape


# In[58]:


data.info()


# In[59]:


data.describe().T


# In[60]:


pd.crosstab(data["job"],data["y"]).plot(kind = "bar")


# In[61]:


pd.crosstab(data["job"],data["y"])


# In[62]:


pd.crosstab(data["job"],data["y"]).sum(1)


# In[63]:


pd.crosstab(data["job"],data["y"]).div(pd.crosstab(data["job"],data["y"]).sum(1), axis =0)


# In[64]:


pd.crosstab(data["job"],data["y"]).div(pd.crosstab(data["job"],data["y"]).sum(1), axis =0).plot(kind = "bar", stacked= True)


# In[65]:


pd.crosstab(data["marital"],data["y"]).div(pd.crosstab(data["marital"],data["y"]).sum(1), axis =0).plot(kind = "bar", stacked= True)


# In[66]:


table = pd.crosstab(data["education"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[67]:


table = pd.crosstab(data["housing"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[68]:


table = pd.crosstab(data["loan"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[69]:


table = pd.crosstab(data["contact"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[70]:



table = pd.crosstab(data["month"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[71]:


table = pd.crosstab(data["day_of_week"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[72]:


table = pd.crosstab(data["campaign"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[73]:


table = pd.crosstab(data["pdays"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[74]:


table = pd.crosstab(data["previous"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# In[75]:


table = pd.crosstab(data["poutcome"],data["y"])
plot_data = table.div(table.sum(1), axis =0)
plot_data.plot(kind = "bar", stacked= True)


# ## Feauture Engineering

# In[76]:


data.columns


# In[77]:


data.dtypes


# In[78]:


sns.heatmap(data.corr())


# In[79]:


data.drop(columns = "previous", inplace = True)


# In[80]:


sns.heatmap(data.corr())


# In[75]:


pandas_profiling.ProfileReport(data)


# In[81]:


data.drop(columns = "euribor3m", inplace = True)


# In[82]:


cat_data = data.select_dtypes(exclude = np.number)


# In[83]:


num_data = data.select_dtypes(include = np.number)


# In[84]:


cat_col_encoded = pd.get_dummies(cat_data)


# In[85]:


data_preprocessed = pd.concat([num_data, cat_col_encoded], axis = "columns")


# ## Modelling

# In[86]:


X = data_preprocessed.drop(columns = "y")


# In[87]:


y = data_preprocessed["y"]


# In[88]:


std = StandardScaler()
X_std = std.fit_transform(X)


# In[89]:


norm = MinMaxScaler()
X_norm = norm.fit_transform(X)


# In[90]:


def roc_draw(X_test, y_test,logreg):
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
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


# In[91]:


def model_fit(model, X, y, roc = False, conf = False, threshold = 0.5):
    train_X, test_X, train_y, test_y =  train_test_split(X, y, test_size = 0.3, random_state=1)
    print(np.array(np.unique(test_y, return_counts=True)).T)
    model.fit(train_X, train_y)
    train_pred = model.predict(train_X)
    print("Train Accuracy : ",accuracy_score(train_pred,train_y))
    print("Train Recall : ",recall_score(train_y, train_pred))
    print("Train Precision : ",precision_score(train_y, train_pred))
    test_pred = model.predict(test_X)
    print("Test Accuracy : ",accuracy_score(test_pred,test_y))
    print("Test Recall : ",recall_score(test_y,test_pred))
    print("Test Precision : ",precision_score(test_y,test_pred))
    if roc:
        roc_draw(test_X, test_y, model)
    if conf:
        print("Test Data Confusion Matrix")
        print(confusion_matrix(test_pred,test_y))
    print("After Tuning Threshold")
    test_pred_prob = model.predict_proba(test_X)
    predict_threshold_test = np.where(test_pred_prob[:,1]>threshold,1,0)
    print("Test Accuracy : ",accuracy_score(predict_threshold_test,test_y))
    print("Test Recall : ",recall_score(test_y, predict_threshold_test))
    print("Test Precision : ",precision_score(test_y, predict_threshold_test))
    if conf:
        print("Test Data Confusion Matrix")
        print(confusion_matrix(predict_threshold_test,test_y))
        print(classification_report(test_y, predict_threshold_test))
    return model.predict_proba(train_X), model.predict_proba(test_X)


# In[92]:


logistic = LogisticRegression()
train_pred_prob, test_pred_prob = model_fit(logistic, X, y, roc = True, conf = True, threshold=0.3)


# In[93]:


predict_threshold_test = np.where(test_pred_prob[:,1]>0.7,1,0)


# In[94]:


data_preprocessed.y.value_counts()


# In[95]:


np.where(logistic.predict_proba(X)[:,1]>0.5,1,0)


# In[96]:


logistic.predict_proba(X)


# In[97]:


model_fit(logistic, X_std, y)


# In[98]:


model_fit(logistic, X_norm, y)


# In[99]:


lasso = LogisticRegression(penalty="l2")
model_fit(lasso, X_std, y)


# In[100]:


knn = KNeighborsClassifier()
model_fit(knn, X_std, y)


# In[101]:


nb = GaussianNB()
model_fit(nb, X_std, y)


# In[102]:


bnb = BernoulliNB()
model_fit(bnb, X_std, y)


# In[103]:


data.y.value_counts()


# In[104]:


import imblearn
imblearn.__version__


# In[105]:


from imblearn.under_sampling import NearMiss


# In[106]:


undersample = NearMiss()


# In[107]:


X_undersample, y_undersample = undersample.fit_sample(data_preprocessed.drop(columns="y"),data_preprocessed["y"])


# In[108]:


print(np.array(np.unique(y_undersample, return_counts=True)).T)


# In[109]:


logistic = LogisticRegression()
train_pred_prob, test_pred_prob = model_fit(logistic, X_undersample, y_undersample, roc = True, conf = True, threshold=0.3)


# In[ ]:


pip install dtreeplt


# In[110]:


from dtreeplt import dtreeplt


# In[ ]:





# In[ ]:





# In[111]:


from sklearn.tree import DecisionTreeClassifier


# In[112]:


dt = DecisionTreeClassifier()


# In[214]:


dt.fit( X_undersample, y_undersample)


# In[117]:


X_undersample_predict = dt.predict(X_undersample)


# In[118]:


accuracy_score(y_undersample, X_undersample_predict)


# In[119]:


train_pred_prob, test_pred_prob = model_fit(dt, X_undersample, y_undersample, roc = True, conf = True, threshold=0.3)


# In[201]:


from dtreeplt import dtreeplt


# In[209]:


y_undersample.head()


# In[215]:


dtree = dtreeplt(
    model=dt,
    feature_names=list(X_undersample.columns), target_names=list('y'))
fig = dtree.view()


# In[120]:


X_undersample.shape


# In[126]:


def model_fit_dt(model, X, y, roc = False, conf = False, threshold = 0.5):
    train_X, test_X, train_y, test_y =  train_test_split(X, y, test_size = 0.3, random_state=1)
    print(np.array(np.unique(test_y, return_counts=True)).T)
    model.fit(train_X, train_y)
    train_pred = model.predict(train_X)
    print("Train Accuracy : ",accuracy_score(train_pred,train_y))
    #print("Train Recall : ",recall_score(train_y, train_pred))
    #print("Train Precision : ",precision_score(train_y, train_pred))
    test_pred = model.predict(test_X)
    print("Test Accuracy : ",accuracy_score(test_pred,test_y))
    #print("Test Recall : ",recall_score(test_y,test_pred))
    #print("Test Precision : ",precision_score(test_y,test_pred))
    if roc:
        roc_draw(test_X, test_y, model)
    if conf:
        print("Test Data Confusion Matrix")
        print(confusion_matrix(test_pred,test_y))
    #print("After Tuning Threshold")
    ##test_pred_prob = model.predict_proba(test_X)
    #predict_threshold_test = np.where(test_pred_prob[:,1]>threshold,1,0)
    #print("Test Accuracy : ",accuracy_score(predict_threshold_test,test_y))
    #print("Test Recall : ",recall_score(test_y, predict_threshold_test))
    #print("Test Precision : ",precision_score(test_y, predict_threshold_test))
    if conf:
        print("Test Data Confusion Matrix")
        print(confusion_matrix(predict_threshold_test,test_y))
        print(classification_report(test_y, predict_threshold_test))
    return accuracy_score(train_pred,train_y), accuracy_score(test_pred,test_y)


# In[121]:


max_depth_check = [5,10,15,20,25,30,35,40,45,50]


# In[131]:


def check_dept(max_depth_check):
    train_acc_list= []
    test_acc_list = []
    for i in max_depth_check:
        print("Max_ Depth ----------------------- = ",i)
        dt = DecisionTreeClassifier(max_depth=i)
        train_acc, test_acc = model_fit_dt(dt, X_undersample, y_undersample, roc = False, conf = False, threshold=0.3)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list


# In[133]:


train_acc, test_acc = check_dept(max_depth_check)


# In[137]:


resut = pd.DataFrame([max_depth_check,train_acc,test_acc]).T


# In[138]:


resut.columns = ["max_depth", "train", "test"]


# In[140]:


resut.plot(x = "max_depth",y=["train","test"])


# In[141]:


max_depth_check = [2,3,4,5,6,7,8,9,10]


# In[142]:


def check_dept(max_depth_check):
    train_acc_list= []
    test_acc_list = []
    for i in max_depth_check:
        print("Max_ Depth ----------------------- = ",i)
        dt = DecisionTreeClassifier(max_depth=i)
        train_acc, test_acc = model_fit_dt(dt, X_undersample, y_undersample, roc = False, conf = False, threshold=0.3)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list


# In[143]:


train_acc, test_acc = check_dept(max_depth_check)


# In[144]:


resut = pd.DataFrame([max_depth_check,train_acc,test_acc]).T


# In[145]:


resut.columns = ["max_depth", "train", "test"]


# In[146]:


resut.plot(x = "max_depth",y=["train","test"])


# In[147]:





# In[ ]:





# In[150]:


type_criterion = ["gini", "entropy"]


# In[151]:


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


# In[152]:


train_acc, test_acc = check_dept(type_criterion)


# In[157]:


resut = pd.DataFrame([type_criterion,train_acc,test_acc]).T


# In[158]:


resut.columns = ["criterion", "train", "test"]


# In[159]:


resut.plot(x = "criterion",y=["train","test"])


# In[160]:


data_preprocessed.head()


# In[176]:


min_samples_split = [20,21,22,23,24,25,26]


# In[177]:


def check_dept(min_samples_split):
    train_acc_list= []
    test_acc_list = []
    for i in min_samples_split:
        print("Minimun Sample split ----------------------- = ",i)
        dt = DecisionTreeClassifier(criterion="gini", max_depth=9,min_samples_split=i)
        train_acc, test_acc = model_fit_dt(dt, X_undersample, y_undersample, roc = False, conf = False, threshold=0.3)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    return train_acc_list, test_acc_list


# In[178]:


train_acc, test_acc = check_dept(min_samples_split)


# In[179]:


resut = pd.DataFrame([min_samples_split,train_acc,test_acc]).T


# In[180]:


resut.columns = ["min_samples_split", "train", "test"]


# In[181]:


resut.plot(x = "min_samples_split",y=["train","test"])


# In[192]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[189]:



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = [
{ 'max_features': [ 0.2, 0.3, 0.4,0.6,0.8,1  ], 'min_samples_split': [18,19,20],
   'max_depth' : [7,8,9]
}
]

tree = DecisionTreeClassifier()

grid_search = GridSearchCV(tree, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_undersample, y_undersample)


# In[190]:


grid_search.best_estimator_


# In[191]:


grid_search.best_score_


# In[ ]:





# In[193]:


grid_search = RandomizedSearchCV(tree, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_undersample, y_undersample)


# In[194]:


grid_search.best_estimator_


# In[ ]:





# In[195]:


from sklearn.ensemble import RandomForestClassifier


# In[218]:


random_classifier = RandomForestClassifier(n_estimators=70)


# In[219]:


random_classifier.fit(X_undersample, y_undersample)


# In[220]:


X_undersample_pred_random = random_classifier.predict(X_undersample)


# In[221]:


accuracy_score(y_undersample, X_undersample_pred_random)


# In[223]:


random_classifier.feature_importances_


# In[224]:


X_undersample.columns


# In[227]:


feature_imp = pd.DataFrame([X_undersample.columns,random_classifier.feature_importances_]).T


# In[228]:


feature_imp.columns = ["columns","importance"]


# In[230]:


feature_imp


# In[237]:


feature_imp.sort_values(by=['importance'],ascending=False, inplace=True)


# In[244]:


feature_imp.head(15).plot(x = "columns", kind = "bar")


# In[253]:


imp_col = feature_imp.head(10)["columns"]


# In[254]:


model_fit(random_classifier,X_undersample[imp_col],y_undersample,roc=True,conf=True)


# In[ ]:





# In[222]:


model_fit(random_classifier,X_undersample,y_undersample,roc= True)


# In[ ]:





# In[216]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = [
{ 
   'n_estimators' : [20, 50, 70, 100, 200]
}
]

tree = RandomForestClassifier()

grid_search = GridSearchCV(tree, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_undersample, y_undersample)


# In[217]:


grid_search.best_estimator_


# In[ ]:





# In[255]:


from sklearn.ensemble import AdaBoostClassifier


# In[256]:


ada = AdaBoostClassifier()


# In[257]:


ada.fit(X_undersample,y_undersample)


# In[258]:


ada_pred = ada.predict(X_undersample)
accuracy_score(y_undersample, ada_pred)


# In[259]:


model_fit(ada,X_undersample, y_undersample, roc=True, conf=True)


# In[ ]:





# In[260]:


n_tree = [40,50,60,70,80,90,100]


# In[261]:


for i in n_tree:
    ada_tune = AdaBoostClassifier(n_estimators=i)
    print("---------------Number of iteration------------ = ",i)
    model_fit(ada_tune, X_undersample,y_undersample,roc=True, conf=True)


# In[ ]:





# In[ ]:


ada.feature_importances_


# In[ ]:





# In[ ]:





# In[262]:


feature_imp = pd.DataFrame([X_undersample.columns,ada.feature_importances_]).T


# In[263]:


feature_imp.columns = ["columns","importance"]


# In[264]:


feature_imp


# In[265]:


feature_imp.sort_values(by=['importance'],ascending=False, inplace=True)


# In[266]:


feature_imp.head(15).plot(x = "columns", kind = "bar")


# In[253]:


imp_col = feature_imp.head(10)["columns"]


# In[267]:


model_fit(ada,X_undersample[imp_col], y_undersample, roc=True, conf=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'max_features': [ 0.2, 0.3, 0.4,0.6,0.8,1  ], 'min_samples_split': [18,19,20],


# In[ ]:





# In[ ]:





# In[ ]:




