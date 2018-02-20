In order to run the code the steps are:

1) run 'adaboost.m', this file executes the adaboost algorithm and 
	find the weights and the best weak learners to use in the test step

	The input are

	Xtr = the training set
	Ytr = the labels
	algo = the type of algorithm to train (it could be 'tree' or 'rls')
	n = the number of weak learners to use, in case of using 'rls' it is fixed to 10
	T = the number of iterations

	The output are

	a = array of the weights related to the chosen classifiers during training
	weak = the array of best chosen classifiers 
	ada_pred = the prediction of adaboost on the training set

	Example: [a, weak, ada_pred] = adaboost2(Xtr, Ytr, 'tree', 10, 100);

	In order to take further information you can run the file with different output:

	err_new = a matrix that contains the errors calculate at each step according to the new distribution
	a = array of the weights related to the chosen classifiers during training
	W = matrix of the different distribution assigned to the sample according to misclassification/correct classification
	R = matrix error - alpha for each chosen weak classifiers 
	H = the evolution of the prediction according to the iterations
	H_f = the same as before but there are errors instead of predictions
	weak = the array of best chosen classifiers weak
	pred = array of predictions of the chosen weak classifiers

2) run 'adaboostTest.m', this file represents the test step

	The input are the same of the previous file, with the addition of the array of weights
	and without the number of weak learners because it is taken from other input.

	The number of iteration is not more useful.

	The output is

	test_pred = the final prediction of adaboost on the test set

	Example: [test_pred] = adaboostTest(Xtr, Ytr, Xts, Yts, a, 'tree', weak);

3) In order to find the best number of iterations you can run 'holdoutCVabT.m'.

	Example: [t, Vm, Vs, Tm, Ts] = holdoutCVabT(X, Y, 0.4, 10, [10 20 50 100 200 500 1000]);