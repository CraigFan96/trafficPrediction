"""
This code is an implementation of the multioutput regressor model that we wrote 
originally to use multioutput regression in a previous version of sklearn learn.
We ended up not using this class and updated sklearn instead.
"""
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

class multioutput:
	def __init__(self, model, num_output_dimensions, params=None):
		self.model = model
		self.num_output_dimensions = num_output_dimensions
		self.model_list = []
		self.params = params

	def fit(self, X,Y):
		for x in range(self.num_output_dimensions):
			if not self.params is None:
				self.model_list.append(GridSearchCV(self.model(), self.params, cv=3, n_jobs=-1).fit(X, Y[:, x]))
			else:
				self.model_list.append(self.model().fit(X, Y[:, x]))

	def predict(self, X):
		output = []
		for x in range(self.num_output_dimensions):
			output.append(self.model_list[x].predict(X))
		return np.swapaxes(np.array(output), 0, 1)

	def score(self,X,Y):
		yhat = []
		for x in range(self.num_output_dimensions):
			yhat.append(self.model_list[x].predict(X))
		return r2_score(np.array(Y), np.swapaxes(np.array(yhat), 0, 1))
		