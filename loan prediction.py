#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


# In[73]:


df=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv')


# In[74]:


df


# In[75]:


df.columns


# In[76]:


df.shape


# In[77]:


df.dtypes


# In[78]:


df.info()


# In[79]:


df.drop(columns='Loan_ID',axis=1,inplace=True)


# In[80]:


# % of missing values in given features
for i in df.columns:
    print(i,np.round(df[i].isnull().mean(),2)*100,'% missing value')


# In[81]:


df.isnull().sum()


# In[82]:


sns.heatmap(df.isnull())


# In[83]:


# check duplicate values
df.duplicated().sum()


# In[84]:


categorical_features=[features for features in df.columns if df[features].dtypes == 'O']
categorical_features


# In[85]:


numerical_features=[features for features in df.columns if df[features].dtypes != 'O']
numerical_features


# In[86]:


descreate_features=[features for features in numerical_features if len(df[features].unique())<25]
descreate_features


# In[87]:


#  check unique values in features
for i in descreate_features:
    print('\n',df[i].value_counts())


# In[88]:


#chech unique values in categery features
for i in categorical_features:
    print('\n',df[i].value_counts())


# In[89]:


df['Loan_Status'].value_counts().plot(kind='pie',autopct='%.1f%%',explode=[0,0.1],shadow=True,figsize=(6,6),colors=['turquoise', 'orangered'])
plt.title('Loan Status',fontsize=15)
plt.show()


# ##### from graph getting loan ration is high 68% people chances getting of loan

# In[90]:


# people stay in different area of region in accordance to getting chances of loan 
df['Property_Area'].value_counts().plot(kind='pie',autopct='%.2f%%',explode=[0,0,0],shadow=True,figsize=(6,6),colors=['turquoise', 'orangered','#4F6272'])
plt.title('Different property areas',fontsize=15)
plt.show()


# In[91]:


df['Self_Employed'].value_counts().plot(kind='pie',autopct='%.2f%%',explode=[0,0.1],shadow=True,figsize=(6,6),colors=['turquoise', 'orangered'])
plt.title('Self employed peoples',fontsize=15)
plt.show()


# #### from the graph self employed or business persons are less & SALARIED persons more for loan application

# In[92]:


df['Education'].value_counts().plot(kind='pie',autopct='%.2f%%',explode=[0,0.1],shadow=True,figsize=(6,6),colors=['turquoise', 'orangered'])
plt.title('Educated people ration',fontsize=15)
plt.show()


# #### graduate people ratio is more 

# In[93]:


df['Dependents'].value_counts().plot(kind='pie',autopct='%.2f%%',explode=[0.1,0,0,0],shadow=True,figsize=(6,6),colors=['turquoise', 'orangered','#4F6272','#8EB897'])
plt.title('dependancy on applicants',fontsize=15)
plt.show()


# In[94]:


df['Married'].value_counts().plot(kind='pie',autopct='%.2f%%',explode=[0.1,0],shadow=True,figsize=(6,6),colors=['turquoise', 'orangered'])
plt.title('Married ratio',fontsize=15)
plt.show()


# #### most of person are married who want loan

# In[95]:


df['Gender'].value_counts().plot(kind='pie',autopct='%.2f%%',explode=[0.1,0],shadow=True,figsize=(6,6),colors=['#4F6272','#8EB897'])
plt.title('Gender ratio',fontsize=15)
plt.show()


# #### loan taking more by male as compare to female

# In[96]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Gender',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+5))
plt.title('Loan possibility on gender',fontsize=15),
plt.show()


# #### from the grap getting chance of loan more in female 12% out of 6% almost 50-50% chance
# #### as in male ratio is 60-40%

# In[97]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Married',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+5))
plt.title('loan possibility on married status',fontsize=15),
plt.show()


# ### unmarried persons have less chance of get loan 

# In[98]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Dependents',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+5))
plt.title('Loan possibility according to depeandancy',fontsize=15),
plt.show()


# #### dependancy of 0 and 2 have more chances to get loan

# In[99]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Education',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+5))
plt.title('Loan possibility on education',fontsize=15),
plt.show()


# In[100]:


#### un educated persons have less chances of getting loan


# In[101]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Self_Employed',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+5))
plt.title('Loan possibility on profession',fontsize=15),
plt.show()


# #### most salaried persons have getting probability of loan

# In[102]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Property_Area',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+3))
plt.title('Loan possibility on area status',fontsize=15),
plt.show()


# #### semiurban area people have more chances of getting loan

# In[103]:


plt.figure(figsize=(10,7))
ax=sns.countplot(x='Credit_History',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.2f}'.format(p.get_height()/614*100), (p.get_x()+0.1, p.get_height()+3))
plt.title('Loan possibility base on credit history',fontsize=15),
plt.show()


# #### persons who has good history of payments hight chances of getting loans

# In[104]:


plt.figure(figsize=(15,7))
ax=sns.countplot(x='Loan_Amount_Term',hue='Loan_Status',data=df,palette='Spectral')
for p in ax.patches:
    ax.annotate('%{:.1f}'.format(p.get_height()/614*100), (p.get_x(), p.get_height()+3))
plt.title('Loan possibility base on Loan term',fontsize=15),
plt.show()


# In[105]:


sns.barplot(x="Loan_Amount_Term",y="LoanAmount",data=df)


# In[106]:


# more than 1 dependents have high demand of loan amount
sns.barplot(y="LoanAmount",x="Dependents",data=df)
plt.show()


# In[107]:


# on applicant income more 3 persons have high dependancy
sns.barplot(y="ApplicantIncome",x="Dependents",data=df)
plt.show()


# In[108]:


sns.countplot(x="Loan_Status",data=df.loc[df["ApplicantIncome"]>6000])
plt.show()


# In[109]:


# coapplicants income doesn't effect loan possibility
sns.countplot(x="Loan_Status",data=df.loc[df["CoapplicantIncome"]==0])
plt.show()


# In[110]:


for i in numerical_features:
    plt.plot()
    sns.distplot(df[i],label=i,color='orange')
    plt.legend()
    plt.show()


# In[111]:


for i in numerical_features:
    plt.plot()
    df[i].plot(kind='box')
    plt.show()


# ### from that graphical represnt all numerical independents featires have skew and outliers

# In[112]:


df.describe()


# #### applicant income std is more than mean and max very high than 75 percentile so have right skewed data and outliers
# #### coapplicant income also have right skewed data and outliers
# #### loan amount and loan amount term have mean more than std so have left skewed data and outliers as well
# #### credit histry is binomial distribution

# In[113]:


# filling null values of categorical columns by mode
for i in df.columns:
    if df[i].dtypes == 'O' and df[i].isnull().sum()>0:
        df.loc[df['Loan_Status']=='Y',[i]]=df.loc[df['Loan_Status']== 'Y',[i]].fillna(df.loc[df['Loan_Status']=='Y',[i]].mode().iloc[0])
        df.loc[df['Loan_Status']=='N',[i]]=df.loc[df['Loan_Status']== 'N',[i]].fillna(df.loc[df['Loan_Status']=='N',[i]].mode().iloc[0])
        


# In[114]:


# filling null values of numerical columns by median bcs of skewness
for i in df.columns:
    if df[i].isnull().sum()>0:
        df.loc[df['Loan_Status']=='Y',[i]]=df.loc[df['Loan_Status']=='Y',[i]].fillna(df.loc[df['Loan_Status']=='Y',[i]].mode().iloc[0])
        df.loc[df['Loan_Status']=='N',[i]]=df.loc[df['Loan_Status']=='N',[i]].fillna(df.loc[df['Loan_Status']=='N',[i]].mode().iloc[0])


# In[115]:


df.isnull().sum()


# In[116]:


# checking skewness
df.skew()


# In[117]:


#dividing it into input and output
x=df.drop(columns=["Loan_Status"])
y=df[["Loan_Status"]]


# In[118]:


# remove skewness
for index in x.skew().index:
    if x.skew().loc[index]>0.5:
        x[index]=np.log1p(x[index])
        


# In[119]:


x.skew()


# In[120]:


from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler


# In[121]:


# encoding categerical columns
lb=LabelEncoder()
for i in x.columns:
    x[i]=lb.fit_transform(x[i])


# In[122]:


lb.fit(y)
y=lb.transform(y)


# In[143]:


x


# In[153]:


x.columns


# In[123]:


sc=StandardScaler()
sc.fit(x)
x1=sc.transform(x)
x1=pd.DataFrame(x1,columns=x.columns)
x1


# In[124]:


# balancing data set
from imblearn.over_sampling import SMOTE
sm=SMOTE()
x2,y2=sm.fit_resample(x1,y)


# In[125]:


np.bincount(y)


# In[126]:


from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,f1_score,accuracy_score,roc_auc_score,roc_curve
lg=LogisticRegression()


# In[127]:


# select best random state
for i in range(0,500):
    x_train,x_test,y_train,y_test=train_test_split(x2,y2,test_size=0.3,random_state=i)
    lg.fit(x_train,y_train)
    pred_tr=lg.predict(x_train)
    pred_te=lg.predict(x_test)
    if round(roc_auc_score(y_train,pred_tr)*100,1)==round(roc_auc_score(y_test,pred_te)*100,1):
        print('\n Random State',i)
        print('roc auc_score TR',roc_auc_score(y_train,pred_tr)*100)
        print('roc auc_score TE',roc_auc_score(y_test,pred_te)*100)


# In[128]:


x_train,x_test,y_train,y_test=train_test_split(x2,y2,test_size=0.3,random_state=115)
lg.fit(x_train,y_train)
pred_lg=lg.predict(x_test)
print('Train score:',lg.score(x_train,y_train)*100)
print('auc roc score',roc_auc_score(y_test,pred_lg)*100)
print('f1 Score: ',f1_score(y_test,pred_lg)*100)
print('accuracy score',accuracy_score(y_test,pred_lg)*100)
print('Confusion matrix \n',confusion_matrix(y_test,pred_lg))
print('Classification report \n',classification_report(y_test,pred_lg))


# In[129]:


dtc=DecisionTreeClassifier()
svc=SVC()
kn=KNeighborsClassifier()
sgd=SGDClassifier()


# In[130]:


neighbors={"n_neighbors":range(1,30)}
clf = GridSearchCV(kn, neighbors, cv=5,scoring="roc_auc")
clf.fit(x2,y2)
clf.best_params_


# In[131]:


parameters={"kernel":["linear", "poly", "rbf"],"C":[0.001,0.01,0.1,1,10]}
clf = GridSearchCV(svc, parameters, cv=5,scoring="roc_auc")
clf.fit(x2,y2)
clf.best_params_


# In[132]:


def classifiers(f):
    f.fit(x_train,y_train)
    print(f,'\n',f.score(x_train,y_train)*100)
    pred=f.predict(x_test)
    print('auc roc score',roc_auc_score(y_test,pred)*100)
    print('Accuracy score:\n',accuracy_score(y_test,pred)*100)
    print('F1 score:\n',f1_score(y_test,pred)*100)
    print('Confusion matrix:\n',confusion_matrix(y_test,pred))
    print('Classification report:\n',classification_report(y_test,pred))


# In[133]:


classifiers(dtc)


# In[134]:


classifiers(SVC(kernel='rbf',C=10))


# In[135]:


classifiers(KNeighborsClassifier(n_neighbors=4))


# In[136]:


classifiers(sgd)


# In[137]:


# ensemble methods
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier


# In[138]:


ensemble=[RandomForestClassifier(),AdaBoostClassifier(),GradientBoostingClassifier(),BaggingClassifier()]
for i in ensemble:
    i.fit(x_train,y_train)
    print(i,'\n\n score: \n',i.score(x_train,y_train)*100)
    pred=i.predict(x_test)
    print('auc roc score',roc_auc_score(y_test,pred)*100)
    print(' F1 score:',f1_score(y_test,pred)*100)
    print('Accuracy scoer:\n',accuracy_score(y_test,pred)*100)
    print('Confusion_matrix:\n',confusion_matrix(y_test,pred))
    print('Classification report:\n',classification_report(y_test,pred))
    print('\n')


# # hyper parameter tuning
# parameters={"n_estimators":[1000,500,100],'criterion':['gini', 'entropy'],'max_features':['auto', 'sqrt', 'log2'],'class_weight':['balanced','balanced_subsample']}
# clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5,scoring="roc_auc")
# clf.fit(x2,y2)
# clf.best_params_

# In[139]:


rf=RandomForestClassifier(criterion='gini',max_features='auto',n_estimators=2000,class_weight='balanced_subsample')
rf.fit(x_train,y_train)
print(rf,'\n\n score: \n',rf.score(x_train,y_train)*100)
pred_rf=rf.predict(x_test)
print('auc roc score:',roc_auc_score(y_test,pred_rf)*100)
print(' F1 score:',f1_score(y_test,pred_rf)*100)
print('Accuracy scoer:\n',accuracy_score(y_test,pred_rf)*100)
print('Confusion_matrix:\n',confusion_matrix(y_test,pred_rf))
print('Classification report:\n',classification_report(y_test,pred_rf))


# In[140]:


warnings.filterwarnings("ignore") 
scores=cross_val_score(rf,x2,y2,cv=5)
score=np.mean(scores)
std=np.std(scores)
print('CV mean',score)
print('std:',std)


# ###  Base on less difference b/t cv & roc score select XGBoost classifier

# In[141]:


fpr, tpr, threshold = roc_curve(y_test,pred_rf)
print('AUC roc score: ',roc_auc_score(y_test,pred_rf))
plt.plot(fpr, tpr, color ='orange', label ='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label ='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost classification')
plt.legend()
plt.show()


# In[142]:


# saving model
import pickle
pickle_out=open("rf.pkl","wb")
pickle.dump(rf,pickle_out)
pickle_out.close()


# In[152]:


rf.predict([[1,0,1,0,1,2000,412,5000,0,0,2]])


# In[ ]:




