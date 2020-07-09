from util import*
import numpy as np

def decision_tree_classifier(self,X,Y):
	from sklearn import tree

	para=self.para['classification']

	clf=tree.DecisionTreeClassifier(
		max_depth=self.call_para('classification','max_depth'),
		min_samples_split=self.call_para('classification','min_samples_split') or 2,
		)
	clf.fit(X,Y)
	return clf

def random_forest_classifier(self,X,Y):
	from sklearn.ensemble import RandomForestClassifier
	para=self.para['classification']

	clf=RandomForestClassifier(
		n_estimators = self.call_para('classification','n_estimators'),
		max_depth = self.call_para('classification','max_depth'),
		min_samples_split = self.call_para('classification','min_samples_split'),
		criterion = self.call_para('classification', 'criterion'),
		min_impurity_decrease = self.call_para('classification', 'min_impurity_decrease'),
		# ccp_alpha = self.call_para('classification', 'ccp_alpha'),
		)

	clf.fit(X,Y)
	return clf


def extreme_forest_classifier(self,X,Y):
	from sklearn.ensemble import ExtraTreesClassifier
	para=self.para['classification']

	clf=ExtraTreesClassifier(
		n_estimators = self.call_para('classification','n_estimators'),
		max_depth = self.call_para('classification','max_depth'),
		min_samples_split = self.call_para('classification','min_samples_split'),
		criterion = self.call_para('classification', 'criterion'),
		)

	clf.fit(X,Y)
	return clf
	
def svm_svc_classifer(self, X, Y):
	from sklearn.svm import SVC

	clf = SVC(
		C = self.call_para('classification', 'reg'),
		gamma = 'auto')

	clf.fit(X, Y)
	return clf

def gaussian_process_classifier(self, X, Y):
	from sklearn.gaussian_process import GaussianProcessClassifier 

	clf = GaussianProcessClassifier()

	clf.fit(X, Y)

	return clf

def neural_network_classifier(self, X, Y):
	from sklearn.neural_network import MLPClassifier 

	clf = MLPClassifier(
		hidden_layer_sizes = self.call_para('classification', 'hidden_layers'),
		alpha = self.call_para('classification', 'alpha'),
		learning_rate = self.call_para('classification', 'learning_rate'),
		activation = self.call_para('classification', 'activation'),
		solver = self.call_para('classification', 'solver'),
		)

	clf.fit(X, Y)

	return clf

def AdaBoost_classifier(self, X, Y):
	from sklearn.ensemble import AdaBoostClassifier

	clf = AdaBoostClassifier(

		)

	clf.fit(X, Y)

	return clf


def QuadraticDiscriminant_classifer(self, X, Y):
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def standard_scaler(self,X):
	from sklearn.preprocessing import StandardScaler
	scaler=StandardScaler()
	scaler.fit(X)
	return scaler

def knn_classifier(self,X,Y):
	from sklearn.neighbors import KNeighborsClassifier
	para=self.para['classification']

	clf=KNeighborsClassifier(n_neighbors=para['n_neighbors'])
	clf.fit(X,Y)

	return clf

def dotpredict_classifier_test(self,clf,X_test,Y_test,X_train,Y_train):

	summary={}
	for i in [(X_test,Y_test,"test"),(X_train,Y_train,"train")]:
		X,Y,name=i
		test_x=X[0]
		Y_pred=clf.predict(X)
		Y_proba=clf.predict_proba(X)

		# for i in np.random.choice(len(Y_pred),20):
		# 	print_debug(f"Test {i}:\n{Y[i]}\n{Y_pred[i]}\n{Y_proba[i]}")

		diff=Y_pred-Y #is non-zero row only when prediction wrong 

		wrongs,total=np.count_nonzero(diff),len(diff)
		trues=total-wrongs
		summary["score "+name]=f"{trues/total*100:.0f}%"

	return summary

def dotscore_classifier_test(self,clf,X_test,Y_test,X_train,Y_train):

	summary={}
	for i in [(X_test,Y_test,"test"),(X_train,Y_train,"train")]:
		X,Y,name=i
		test_x=X[0]
		summary["score "+name]=f"{clf.score(X,Y)*100:.0f}%"

	return summary