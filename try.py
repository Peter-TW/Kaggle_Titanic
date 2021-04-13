import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import re

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#Goal:  find patterns in train.csv that help us predict whether the passengers in test.csv survived.

#Get data
test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")

# Merge the data and clean data together
full_data = pd.concat([test_data,train_data])
# print(full_data.info())
# print(full_data.head())
# print(full_data.describe(include=["O"]))
# #Any missing data to be correcting
MissingCount=full_data.isnull().sum().sort_values(ascending=False)
# # Cabin > Age should clear them


# # CLEAN DATA
# #Cabin have more incomplete data could consider drop it
# #each person have the ticket, the ticket are all uniqe not related to survived so drop it
full_data=full_data.drop(["Cabin","Ticket"],axis=1)
#
# #SEX
# sex to 0 and 1
sex_convert = {"male":1,"female":0}
full_data["Sex"]=full_data["Sex"].map(sex_convert)
#
# #NAME
# # Name to Ttile to 1~5
# # drop  name, PassengerId is linked to name could ignord or drop it
full_data["Name"]=full_data["Name"].str.split(", ",expand=True)[1]
full_data["Title"] =full_data["Name"].str.split(". ",expand=True)[0]
full_data["Title"]=full_data["Title"].replace(['Don','th','Rev','Dr','Major','Lady','Sir','Col','Capt','Countess','Jonkheer','Dona'], 'Rare' )
full_data["Title"]=full_data["Title"].replace(['Ms','Mlle'], 'Miss')
full_data["Title"]=full_data["Title"].replace('Mme', 'Mrs')
title_list = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}
full_data["Title"] = full_data["Title"].map(title_list)
full_data=full_data.drop("Name",axis=1)
#
# #EMABARKED
# #Emabarked have 2 missing, place them to most common embarked
most_in_emabarked = full_data.Embarked.mode()[0]
full_data["Embarked"]= full_data["Embarked"].fillna(most_in_emabarked)
# change SCQ to 123
change_cembarked = {"S":1,"C":2,"Q":3}
full_data["Embarked"]=full_data["Embarked"].map(change_cembarked)
#
# #AGE & TITLE
# #Age around 25 % is missing
# full_data.Age.isna().sum()/full_data.Age.notnull().sum() *100
#Observe age relation in Sex and Pclass in graph and figure
# #lots missing is in Pclass 3, male
full_data["Age_missing"]=full_data["Age"].isnull().map(lambda x:0 if x==True else 1) #create new column if the date missing is 0 else 1
# sns.countplot(full_data.Sex,hue=full_data.Age_missing)# male missing is more than female
# pd.crosstab(full_data.Age_missing,full_data.Sex)

# plt.show()
# sns.countplot(full_data.Age_missing,hue=full_data.Pclass)# Pclass in 3 have more missing than others
# pd.crosstab(full_data.Age_missing,full_data.Pclass) #  missing data in Pclass 3 around 80%  208/263

# # #detail Age in Survived in Pclass in 1&2
# P_Survived = ((full_data.Age_missing == 1) & (full_data.Pclass != 3) & (full_data.Survived ==1))
# P_Dead = ((full_data.Age_missing == 1) & (full_data.Pclass != 3) & (full_data.Survived ==0))
# ax=sns.distplot(full_data.loc[P_Survived ,"Age"],bins=45,label="Survived")
# ax=sns.distplot(full_data.loc[P_Dead,"Age"],bins=45,label="Dead")
# plt.title( 'Pclass not in 3' )
# plt.legend()

#Age <17 (around) have higher survived rate, over 70 have less data and age and title have highly related
# #Use median to replace the missing age of title
# #Only use the data Age under 17 to predicted the survived rate
TMD=full_data.groupby("Title")["Age"].median().values #median to replace
full_data["New_Age"]=full_data["Age"]
for i in range(0,5):
    full_data.loc[(full_data.Age.isnull()) & (full_data.Title == i),"New_Age"]=TMD[i] #
full_data.New_Age=full_data.New_Age.astype("int")
full_data["New_Age_title"] =(full_data.New_Age <17)*1 #devide to 2 group under 17 is 1,over is 0
full_data=full_data.drop(["Age","New_Age","Title","Age_missing"],axis=1)# the data won't use age and title

# #SIBSP, PARCH to Family
# # SibSp and Parch are family size,merge them as family_size and take a look in graph and figure
full_data["Family_size"]=full_data["SibSp"]+full_data["Parch"]+1
# #Graph show 2~4 family members have higher survived rate
# sns.factorplot(x="Family_size",y="Survived",data=full_data)

# #devived them to 3 group,1 person;2~4 menembers;over 4members, name Family group
# full_data["Family_group"]=np.nan
full_data.loc[full_data["Family_size"]==1,"Family_group"]=0
full_data.loc[((full_data["Family_size"]<=4) & (full_data["Family_size"] !=1)),"Family_group"]=1
full_data.loc[full_data["Family_size"]>=5,"Family_group"]=2
full_data.Family_group=full_data.Family_group.astype(int)
#detail in survived data
# full_data[["Family_group","Survived"]].groupby("Family_group").mean().round(4)*100#family group in 1 nearly 58% survived
#drop Family_size,SibSp and Parch in favor of Family_group
full_data=full_data.drop(["Family_size","SibSp","Parch"],axis=1)


# #FARE
# #Fare is one missing fill it with median
full_data.Fare=full_data.Fare.fillna(full_data.Fare.median())
# #Convert Fare to 4 group (Quartile)
full_data["Fare_group"]=pd.qcut(full_data.Fare,4)
# #Fare higher have higher survived rate
# full_data[["Fare_group","Survived"]].groupby("Fare_group").mean().round(4)*100
#asign them to group base on Fare_group
full_data.loc[full_data["Fare"]<=7.896,"Fare"]=0
full_data.loc[(full_data["Fare"]>7.896) & (full_data["Fare"]<=14.454),"Fare"]=1
full_data.loc[(full_data["Fare"]>14.454) & (full_data["Fare"]<=31.275),"Fare"]=2
full_data.loc[full_data["Fare"]>31.275,"Fare"]=3
full_data.Fare=full_data.Fare.astype(int)
full_data=full_data.drop("Fare_group",axis=1)
#
# #SURVIVED to OTHERS
# #FARE;SEX;EMABARKED;PCLASS;NEW_AGE_TITLE;FAMILY_GROUP
full_data[["Fare","Survived"]].groupby("Fare").mean().round(4)*100 #Higher fare higher survived rate
full_data[["Sex","Survived"]].groupby("Sex").mean().round(4)*100#Female survived raate is higher than man
full_data[["Embarked","Survived"]].groupby("Embarked").mean().round(4)*100# Embarked from C have higher survived rate
full_data[["New_Age_title","Survived"]].groupby("New_Age_title").mean().round(4)*100#who over 17 is higher survived rate
full_data[["Family_group","Survived"]].groupby("Family_group").mean().round(4)*100#family group 1 (size 2~4) have higher rate

# #Seperate the train and test data
trainD=full_data[pd.notnull(full_data.Survived)]
testD=full_data[pd.isnull(full_data.Survived)]
testD=testD.drop("Survived",axis=1)
# # print(testD.head())
#Set and lebal the traing data
x_train = trainD.drop(["PassengerId","Survived"],axis=1)
y_train =trainD["Survived"]
# # print(testD.head())
# Decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train,y_train)
acc_decision_tree = round(decision_tree.score(x_train,y_train)*100,2)

#Ramdom forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train,y_train)
acc_random_forest = round(random_forest.score(x_train,y_train)*100,2)

# # print(acc_random_forest)
#Submit
xtest=testD.drop("PassengerId", axis=1)
minor_pred=decision_tree.predict(xtest)
submission = pd.DataFrame({ "PassengerId":testD["PassengerId"],"Survived": minor_pred.astype(int)})
submission.to_csv("Sub2.csv",index=False)
# print(submission.head())
#
#
# # # print(full_data.head())
# # # Relationship between Survived and others
# # #PassengerId is most not related drop it
# # corrall = full_data.corr()
# # corrsurvived = corrall.loc["Survived"].sort_values()[:-1]
# # corrsurvived=pd.DataFrame({"Survived":corrsurvived})
# # full_data.drop("PassengerId",inplace=True,axis=1)
# # # print(corrsurvived)
# #
# # #Create graph and percentage to compare
# # select_list = ["Pclass","SibSp","Parch","Sex","Embarked"]
# # # gs = gridspec.GridSpec(len(select_list),1)
# # # for i, col in enumerate( full_data[select_list]):
# # #     ax = plt.subplot(gs[i])
# # #     sns.countplot(full_data[col],hue=full_data.Survived)
# # #     ax.legend(loc=1)
# # # plt.show()
# # for i in full_data[select_list]:
# #     l=["Survived"]
# #     l.append(i)
# #     Survived_rate = full_data[l].groupby(i).mean().round(3).reset_index()
# #     Survived_rate.columns = [i,"Survived Rate"]
# #     Survived_rate["Survived Rate"]=Survived_rate["Survived Rate"].map(lambda x:x*100)
# #     # print( Survived_rate)
# #
# # #Female have higher survived rate than male, in pclass could be more detail
# # Sexclass_rate = full_data[["Sex","Pclass","Survived"]].groupby(["Sex","Pclass"]).mean()*100
# # # Female in 1,2 class are over 90% will survived
# # # Devivded them to three level (1~3) 3 is highest survived rate name Sex_pclass
# # full_data["Sex_pclass"] = np.nan
# # # full_data.loc["Sex"==1 and "Pclass" == 1]
# #
# # #SibSp and Parch could merge together as family data
# # #SibSp with 1  is higher rate to survived
# # # (Parch) have parents / children 1~3 is higher rate to survived
# # #merge together and see the rate in Survived
# # # full_data["Family_unm"]=full_data.SibSp+full_data.Parch+1 #0 is single and count
# # # famil_sur = full_data[["Family_unm","Survived"]].groupby(by=["Family_unm"]).mean()*100
# # # familnum in 4 is highest survived
# # # print(full_data["Family_unm"].value_counts())
# # # print(famil_sur)