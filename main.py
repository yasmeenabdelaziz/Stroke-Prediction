import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn import svm
import matplotlib.pyplot as plt

HealthCare = pd.read_csv("healthcare-dataset-stroke-data.csv")
HealthCare = HealthCare.iloc[:, 1:]

le = LabelEncoder()
gender = le.fit_transform(HealthCare['gender'])
HealthCare['gender'] = gender

ever_married = le.fit_transform(HealthCare['ever_married'])
HealthCare['ever_married'] = ever_married

work_type = le.fit_transform(HealthCare['work_type'])
HealthCare['work_type'] = work_type

Residence_type = le.fit_transform(HealthCare['Residence_type'])
HealthCare['Residence_type'] = Residence_type

smoking_status = le.fit_transform(HealthCare['smoking_status'])
HealthCare['smoking_status'] = smoking_status

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(HealthCare)
healthcare = imputer.transform(HealthCare)

HealthCare = pd.DataFrame(healthcare,
                          columns=['gender', "age", "hypertension", "heart_disease", "ever_married", "work_type",
                                   "Residence_type", "avg_glucose_level", "bmi", "smoking_status", "stroke"])

HealthCare=HealthCare.drop_duplicates()
HealthCare = HealthCare.drop('work_type', 1)
HealthCare = HealthCare.drop('ever_married', 1)
HealthCare = HealthCare.drop('Residence_type', 1)
dataTrain = HealthCare.iloc[0:4088, :]
dataTest = HealthCare.iloc[4088:, :]
Data = HealthCare.iloc[:, 1:7].values
Target = HealthCare['stroke']
X_train, X_test, Y_train, Y_test = train_test_split(Data, Target, test_size=0.2, random_state=0)


def ID3():
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_train, Y_train)
    Y_predict = decision_tree_model.predict(X_test)
    print("T1_TREE:")
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_predict))
    clf = DecisionTreeClassifier()
    clf = clf.fit(dataTrain.iloc[:, 1:7], dataTrain["stroke"])
    res = clf.predict(dataTest.iloc[:, 1:7])
    print("T2_TREE:")
    print("ID3_Result : ", res)
    acc = accuracy_score(dataTest["stroke"], res)
    print("ID3_Accuracy : ", acc)
    print("ID3_Error : ",1-acc)
    #print("Tree : ", plot_tree(clf))


def KNN():
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, Y_train)
    Y_predict = neigh.predict(X_test)
    print("K1_KNN:")
    print("KNN_Accuracy :", metrics.accuracy_score(Y_test, Y_predict))

    neigh.fit(dataTrain.iloc[:, 1:7], dataTrain["stroke"])
    res = neigh.predict(dataTest.iloc[:, 1: 7])
    print("K2_KNN:")
    print("KNN_Result : ", res)
    accuracy = accuracy_score(dataTest["stroke"], res)
    print("KNN_Accuracy : ", accuracy)
    print("KNN_Error : ", 1 - accuracy)

    #print("KNN Graph:")
    #plt.scatter(Y_test, Y_predict)
    #plt.show()


def GNB():
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_predict = gnb.predict(X_test)
    print("G1_GNB:")
    print("GNB_Accuracy : ", metrics.accuracy_score(Y_test, Y_predict))

    gnb.fit(dataTrain.iloc[:, 1:7], dataTrain["stroke"])
    res = gnb.predict(dataTest.iloc[:, 1:7])
    print("G2_GNB:")
    print("GNB_Result : ", res)
    accuracy = accuracy_score(dataTest["stroke"], res)
    print("GNB_Accuracy : ", accuracy)
    print("GNB_Error : ", 1-accuracy)
    #print("GNB Graph:")
    #plt.bar(Y_test, Y_predict)
    #plt.show()


#def svmClassify():
    #classify = svm.SVC(kernel='linear')
    #classify.fit(X_train, Y_train)
    #y_predict = classify.predict(X_test)
    #Accuracy = metrics.accuracy_score(Y_test, y_predict)
    #print("S1_SVM:")
    #print("Accuracy:", Accuracy)
    #classify.fit(dataTrain.iloc[:, 1:7], dataTrain["stroke"])
    #y_predict = classify.predict(dataTest.iloc[:, 1:7])
    #Accuracy = metrics.accuracy_score(dataTest["stroke"], y_predict)
    #print("S2_SVM:")
    #print("Accuracy:", Accuracy)

    #print("SVM Graph:")
    #plt.scatter(Y_test, y_predict)
    #plt.show()


print("--------------Naïve Bayes Model--------------")
GNB()
print("\n--------------Decision Tree Model------------")
ID3()
print("\n--------------KNN Model----------------------")
KNN()
#print("\n--------------SVM Model----------------------")
#svmClassify()