import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def classify(normalize=False, enablePCA=False, PCAAccuracy=.99):
	print('* normalize:\t' + str(normalize))
	print('* enable PCA:\t' + str(enablePCA))
	if enablePCA:
		print('* PCA Accuracy:\t' + str(PCAAccuracy))
	print()

	with open('data/train_labels.csv') as labelsFile:
		trainLabels = pd.read_csv(labelsFile)

	with open('data/train_Comb.csv') as featuresFile:
		trainComb = pd.read_csv(featuresFile)

	mydata = trainComb.drop('Id', axis=1)  # Separating out the data

	if normalize:
		min_max_scaler = preprocessing.MinMaxScaler()
		np_scaled = min_max_scaler.fit_transform(mydata)
		mydata = pd.DataFrame(np_scaled)

	labels = trainLabels["Class"].values
	features = [data.values.tolist() for ind, data in mydata.iterrows()]

	# Initializing Gaussian Naive Bayes classifier
	clfGNB = GaussianNB()

	# Initializing K-Nearest Neighbours classifier
	clfKNN = KNeighborsClassifier(n_neighbors=10)

	# Initializing Random Forest classifier
	clfRFAuto = RandomForestClassifier(n_estimators=1000)
	clfRFSqrt = RandomForestClassifier(n_estimators=1000, max_features="sqrt")
	clfRFLog = RandomForestClassifier(n_estimators=1000, max_features="log2")

	# Initializing Decision Tree classifier
	clfDT = DecisionTreeClassifier()

	# Initializing Support vector machines (SVM) classifiers
	clfSvmLinear = SVC(kernel='linear', C=2)
	clfSvmPoly = SVC(kernel='poly', C=2)
	clfSvmSigmoid = SVC(kernel='sigmoid', C=2)
	clfSvmRbf = SVC(kernel='rbf', gamma=0.01, C=2)

	classifierList = [
		clfKNN,
		clfGNB,
		clfRFAuto,
		clfRFSqrt,
		clfRFLog,
		clfDT,
		clfSvmLinear,
		clfSvmPoly,
		clfSvmSigmoid,
		clfSvmRbf,
	]
	classifiernames = [
		'K Nearest Neighbours',
		'Gaussian Bayes',
		'Random Forest all features',
		'Random Forest max_features=sqrt',
		'Random Forest max_features=log2',
		'Decision Tree',
		'SVM with Linear kernel',
		'SVM with Polynomial kernel',
		'SVM with Sigmoid kernel',
		'SVM with Rbf kernel',
	]
	leaveOneOut = LeaveOneOut()

	for index, estimator in enumerate(classifierList):
		hit = 0
		print('-\t' + classifiernames[index])
		for train_index, test_index in leaveOneOut.split(np.array(mydata)):
			X_train, X_test = np.array(features)[train_index], np.array(features)[test_index]
			y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
			if enablePCA:
				scaler = StandardScaler()

				# Fit on training set only.
				scaler.fit(X_train)
				X_train = scaler.transform(X_train)

				# Make an instance of the Model
				pca = PCA(PCAAccuracy)
				pca.fit(X_train)
				# print("\tNumber of minimum number of principal components such that 99% of the variance is retained is", format(enablePCA.n_components_))

				# Apply the mapping (transform)
				X_train = pca.transform(X_train)
				X_test = pca.transform(X_test)

			clfFit = estimator.fit(X_train, y_train)
			hit += clfFit.score(X_test, y_test)
		print('\tScore: ' + "{0:.2f}".format(hit / len(mydata) * 100) + '\n')


if __name__ == '__main__':
	if len(sys.argv) == 1:
		classify()
	elif len(sys.argv) == 2:
		classify(normalize=sys.argv[1] == 'True')
	elif len(sys.argv) == 3:
		classify(normalize=sys.argv[1] == 'True', enablePCA=sys.argv[2] == 'True')
	elif len(sys.argv) == 4:
		classify(normalize=sys.argv[1] == 'True', enablePCA=sys.argv[2] == 'True', PCAAccuracy=float(sys.argv[3]))
	else:
		print('python ' + sys.argv[0] + '<normalize> <enable PCA> <PCA Accuracy>')
		print('normalise: True/False')
		print('enable PCA: True/False')
		print('PCA Accuracy: float in range of [0, 1]')
