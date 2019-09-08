import numpy as np
__author__="Giuliano Oliveira"
class LinearRegression():
	def __init__(self,*args,alpha=None,iv=None,normalize=False):
		"""
		LinearRegression(x_0,x_1 ... x_n, y, alpha, iv)
		if iv is not set, then it will be initialized with 0
		normalize indicates if the algorithm should normalize the X's on input
		"""
		assert len(args)>=2
		assert alpha!=None
		for vect in args:
			assert vect.shape==args[0].shape
		self.alpha=alpha
		self.m=len(args[0])
		self.Y=args[-1].reshape(self.m,1)
		args=args[:-1]
		self.n=len(args)
		self.X=np.ones((self.m,self.n+1))
		for i in range(self.n):
			if normalize:
				normalize_vec(args[i])
			self.X[:,i+1]=args[i]
		if iv==None:
			self.theta=np.array([0.]*(self.n+1)).reshape((self.n+1,1))
		else:
			assert iv.shape==(self.m,1)
			self.theta=iv
	def __iter__(self):
		return self
	def __next__(self):
		"""Step in iteration, returns curr theta and j"""
		H_theta=self.X.dot(self.theta)
		e=H_theta-self.Y
		j=float(e.T.dot(e)/(2*self.m))
		self.theta-=(self.alpha/self.m)*(self.X.T.dot(e))
		return self.theta.reshape(self.n+1),j
def normalize_vec(X):
	assert len(X.shape)==1
	n=len(X)
	mu=np.mean(X)
	std_dev=np.sqrt(sum(((x-mu)**2 for x in X))/n)
	for i in range(n):
		X[i]=(X[i]-mu)/std_dev

