# This code creates the machine learning models for the detection of fatigue
# This code calls MyDat and AkiDat scripts to collect the extracted features from the physiological signals.
# MyDat and AkiDat are scripts written to process the physiological data and extract the features.
# BestModel3 script is created to perform Meta-Learning technique and provide the best machine learning model to detect the different types of fatigue.

from MyDat import MyDat
from AkiDat import AkiDat
import numpy as np
from sklearn.utils import shuffle
import BestModel3 as BestModel
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import VotingClassifier


# Extract the data
ABK0, ABK2, ABASE = AkiDat()

Base, B_0, A_0, Bk_2 = MyDat()

Rel_Lab_a = np.zeros((np.size(ABK0, 0),), dtype=int)
Phys_Lab_a = np.full((np.size(ABK2, 0),), 1, dtype=int)

Lab_a = np.append(Rel_Lab_a,Phys_Lab_a)
A_Dat = np.vstack((ABK0-ABASE,ABK2-ABASE))

nums = 10
cls = SelectKBest(k=nums) # For Parameter Reduction
A_Dat1 = cls.fit_transform(A_Dat, Lab_a)

cols = cls.get_support()




Rel_Lab_m = np.zeros((np.size(B_0, 0),), dtype=int)
Phys_Lab_m = np.full((np.size(Bk_2, 0),), 1, dtype=int)

Lab_m_1 = np.append(Rel_Lab_m,Phys_Lab_m)
M_Dat_1 = np.vstack((B_0-Base,Bk_2-Base))


M_Dat_1 = cls.transform(M_Dat_1)

A_Dat2 = np.vstack((A_Dat1,M_Dat_1[:5,:]))
A_Dat2 = np.vstack((A_Dat2,M_Dat_1[-5:,:]))

Lab_a_1 = np.append(Lab_a, Lab_m_1[:5])
Lab_a_1 = np.append(Lab_a_1, Lab_m_1[-5:])

M_Dat_2= M_Dat_1[5:-6,:]
Lab_m = Lab_m_1[5:-6]


scalar = MinMaxScaler() # Transforms the data between 0 and 1
A_Dat3 = scalar.fit_transform(A_Dat2)

M_Dat = scalar.transform(M_Dat_2)


M_Dat = M_Dat_1
Lab_m = Lab_m_1
X, y = shuffle(A_Dat1,Lab_a)

result_RF, result_RF_ada = BestModel.RF_Best(X,y)
result_NN, result_NN_ada = BestModel.NN_Best(X,y)
result_KNN, result_KNN_ada = BestModel.Knn_Best(X,y)
result_GB = BestModel.GB_Best(X,y)
result_HB = BestModel.HB_Best(X,y)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
space = dict()
space['voting'] = ['hard', 'soft']
space['flatten_transform'] = [True, False]

# Create a Bagging Classifier of different types of weaker classifier
clf = VotingClassifier(estimators=[('RF', result_RF), ('NN', result_NN),
                                   ('KNN', result_KNN), ('GB', result_GB), ('HB', result_HB)],
                       n_jobs=-1)

clf_ada = VotingClassifier(estimators=[('RF', result_RF_ada), ('NN', result_NN_ada),
                                   ('KNN', result_KNN_ada), ('GB', result_GB), ('HB', result_HB)],
                           n_jobs=-1)

search = RandomizedSearchCV(clf, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
search_ada = RandomizedSearchCV(clf_ada, space, n_iter=500, scoring='f1_micro', n_jobs=-1, cv=cv, random_state=1,verbose=10)
# execute search
result_vot = search.fit(X, y)
result_vot_ada = search_ada.fit(X, y)

# Print the accuracy, F1 Score and the confusion matrix for each classifier
res1 = result_RF.best_estimator_.predict(M_Dat)
print('RF Org: ' + str(result_RF.best_score_))
print('RF Acc: ' + str(metrics.accuracy_score(res1,Lab_m)) + ' F1: ' + str(metrics.f1_score(res1,Lab_m)))
print('RF:\n '+str(metrics.confusion_matrix(Lab_m,res1)))

res2 = result_NN.best_estimator_.predict(M_Dat)
print('NN Org: ' + str(result_NN.best_score_))
print('NN Acc: ' + str(metrics.accuracy_score(res2,Lab_m)) + ' F1: ' + str(metrics.f1_score(res2,Lab_m)))
print('NN:\n '+str(metrics.confusion_matrix(Lab_m,res2)))

res3 = result_KNN.best_estimator_.predict(M_Dat)
print('KNN Org: ' + str(result_KNN.best_score_))
print('KNN Acc: ' + str(metrics.accuracy_score(res2,Lab_m)) + ' F1: ' + str(metrics.f1_score(res2,Lab_m)))
print('KNN:\n '+str(metrics.confusion_matrix(Lab_m,res3)))


res5 = result_RF_ada.best_estimator_.predict(M_Dat)
print('RF ADA Org: ' + str(result_RF_ada.best_score_))
print('RF ADA Acc: ' + str(metrics.accuracy_score(res5,Lab_m)) + ' F1: ' + str(metrics.f1_score(res5,Lab_m)))
print('RF ADA:\n '+str(metrics.confusion_matrix(Lab_m,res5)))

res6 = result_NN_ada.best_estimator_.predict(M_Dat)
print('NN ADA Org: ' + str(result_NN_ada.best_score_))
print('NN ADA Acc: ' + str(metrics.accuracy_score(res6,Lab_m)) + ' F1: ' + str(metrics.f1_score(res6,Lab_m)))
print('NN ADA:\n '+str(metrics.confusion_matrix(Lab_m,res6)))

res7 = result_KNN_ada.best_estimator_.predict(M_Dat)
print('KNN ADA Org: ' + str(result_KNN_ada.best_score_))
print('KNN ADA Acc: ' + str(metrics.accuracy_score(res7,Lab_m)) + ' F1: ' + str(metrics.f1_score(res7,Lab_m)))
print('KNN ADA:\n '+str(metrics.confusion_matrix(Lab_m,res7)))



res8 = result_GB.best_estimator_.predict(M_Dat)
print('GB Org: ' + str(result_GB.best_score_))
print('GB Acc: ' + str(metrics.accuracy_score(res8,Lab_m)) + ' F1: ' + str(metrics.f1_score(res8,Lab_m)))
print('GB:\n '+str(metrics.confusion_matrix(Lab_m,res8)))

res9 = result_HB.best_estimator_.predict(M_Dat)
print('HB Org: ' + str(result_HB.best_score_))
print('HB Acc: ' + str(metrics.accuracy_score(res9,Lab_m)) + ' F1: ' + str(metrics.f1_score(res9,Lab_m)))
print('HB:\n '+str(metrics.confusion_matrix(Lab_m,res9)))

res10 = result_vot.best_estimator_.predict(M_Dat)
print('Vote Acc: ' + str(metrics.accuracy_score(res10,Lab_m)) + ' F1: ' + str(metrics.f1_score(res10,Lab_m)))
print('Vote:\n '+str(metrics.confusion_matrix(Lab_m,res10)))

res11 = result_vot_ada.best_estimator_.predict(M_Dat)
print('Vote ADA Acc: ' + str(metrics.accuracy_score(res11,Lab_m)) + ' F1: ' + str(metrics.f1_score(res11,Lab_m)))
print('Vote ADA:\n '+str(metrics.confusion_matrix(Lab_m,res11)))


print('done')