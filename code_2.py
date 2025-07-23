import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,f1_score,roc_auc_score
from imblearn.combine import SMOTETomek

df= pd.read_csv('KaggleV2-May-2016.csv')
print(df.info())

df['ScheduledDay']=pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay']=pd.to_datetime(df['AppointmentDay'])
df['No-show']=df['No-show'].map({'Yes':1,'No':0})

df.columns=df.columns.str.lower().str.replace('-','_')
df['scheduled_weekday']=df['scheduledday'].dt.dayofweek
df['appointment_weekday']=df['appointmentday'].dt.dayofweek
df['waitingdays']=(df['appointmentday']-df['scheduledday']).dt.days
df['sameday']=(df['waitingdays']==0).astype(int)
df['age_bucket']=pd.cut(df['age'],bins=[0,18,40,60,100],labels=['child','adult','middleaged','senior'])
no_show_rate=df.groupby('neighbourhood')['no_show'].mean()
df['neighbourhood_nsrate']=df['neighbourhood'].map(no_show_rate)
df['appointment_count']=df['patientid'].map(df['patientid'].value_counts())

le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['age_bucket']=le.fit_transform(df['age_bucket'])

features=['age','gender','scholarship','hipertension','diabetes','alcoholism','handcap','sms_received','age_bucket','sameday','scheduled_weekday','appointment_weekday','waitingdays',
          'neighbourhood_nsrate','appointment_count']
X=df[features]
y=df['no_show']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

smote=SMOTETomek(random_state=42)
X_r,y_r=smote.fit_resample(X_train,y_train)

para_grid={
    'n_estimators':[100,200],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1],
    'scale_pos_weight':[1,2,4]
}

model = XGBClassifier( random_state=42, eval_metric='logloss')

grid=GridSearchCV(estimator=model,
                  param_grid=para_grid,
                  scoring='f1_macro',
                  cv=3,
                  verbose=1,
                  n_jobs=-1)
grid.fit(X_r,y_r)
print("best parameters:",grid.best_params_)
best_model=grid.best_estimator_


y_pred=best_model.predict(X_test)
cr=classification_report(y_test,y_pred)
ar=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,average='macro')
print("Classification Report:\n",cr)
print("Accuracy Score:",ar)
print("f1 score",f1)
ra=roc_auc_score(y_test,grid.best_estimator_.predict_proba(X_test)[:,1])
print("ROC AUC:",ra)



