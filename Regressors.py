# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np, scipy as sp
from math import sqrt, exp, ceil, floor , isclose
from sklearn import cluster, linear_model, ensemble, preprocessing, svm, kernel_ridge, metrics

###############################
# Tool functions
###############################

# function that looks like a call option profile
def callLike(x,shift=0.0,factor=1.0):
    return np.log( 1.0 + np.exp(factor*(x-shift)) ) / factor

# function that looks like a call spread option profile
def callSpreadLike(x,shift=0.0,factor=1.0):
    return np.log( 1.0 + np.exp(factor*(x-shift)) ) / factor

# check if a payoff has a cap or a floor, triggers are % of empirical distribution that validate the presence of a limit
def checkPayoffCapFloor( payoff, floorTrigger = 0.0001, capTrigger = 0.0001):

    if payoff is None or payoff.shape[0]==0:
        return False, None, False, None
    
    nbSamples = payoff.shape[0]
    
    floor = np.amin(payoff)
    cap = np.amax(payoff)
    
    floorOccurences = np.sum(np.isclose(payoff,floor)) #/ nbSamples
    capOccurences = np.sum(np.isclose(payoff,cap)) #/ nbSamples
    
    checkHasFloor = ( floorOccurences >= 2 )
    checkHasCap = ( capOccurences >= 2 )
    
    return checkHasFloor, floor if checkHasFloor else None, checkHasCap, cap if checkHasCap else None


###############################
# Regressor classes
###############################

# Simple logistic classifier
class LogisticClassifier(object):
    
    def __init__(self,C=1.0,d=2):
        
        self.C = C
        self.d = d

        # initialise the polynomial regressor
        self.polynomialTransformer = preprocessing.PolynomialFeatures(self.d,False,False)
        self.clf = linear_model.LogisticRegression(C=1.0)

    def fit(self,X,y):        
        data = self.polynomialTransformer.fit_transform(X)
        self.clf.fit(data,y)
        return self
            
    def predict_proba(self,X):
        data = self.polynomialTransformer.fit_transform(X)
        return self.clf.predict_proba(data)


###############################
# Combination of projection on factor calls or moments with regression on remainder
###############################

from Payoffs import CallPrice
normalMoments = [0,1,0,3,0,15]

def getFactorMoment(Z,T,d):
    
    if d==1:
        return Z
    elif d==2:
        return Z**2 + T
    elif d==3:
        return Z**3 + 3.*Z*T
    elif d==4:
        return Z**4 + 6.*(Z**2)*T + 3.*(T**2)
    elif d==5:
        return Z**5 + 10.*(Z**3)*T + 15.*Z*(T**2)
    elif d==6:
        return Z**6 + 15.*(Z**4)*T + 45.*(Z**2)*(T**2) + 15.*(T**3)
    else:
        raise Exception('Factor moment degree must be between 1 and 6')


def getFactorProjection(projectionType,projectionParams,T,Z):
    
    if projectionParams is None:
        return np.zeros((Z.shape[0]))
    
    factorProjectors = None
    if projectionType == 'calls':
    
        # compute all the calls on factor + fwd
        factorProjectors = np.empty((Z.shape[0],projectionParams['strikes'].shape[1]+1))
        factorProjectors[:,0] = Z.reshape((-1))
        for idx, strike in enumerate(projectionParams['strikes'][0,:]):
            factorProjectors[:,idx+1] = CallPrice(strike,Z.reshape((-1)),T,'n',1.0)
            
    else:
        
        # compute all the moments of factor
        factorProjectors = np.empty((Z.shape[0],projectionParams['degree']))
        for i in range(projectionParams['degree']):
            factorProjectors[:,i] = getFactorMoment(Z.reshape((-1)),T,i+1)

    # then use embedded regressor to compute the projection
    return projectionParams['decomposition'].predict(factorProjectors)


# Regressor recomposed from factor calls or moments projection and remainder regressor
class ProjectionRegressor(object):
    
    def __init__(self,projectionType,projection,remainderRegressor):
        super().__init__()
        
        self.projectionType = projectionType
        self.projection = projection
        self.remainderRegressor = remainderRegressor
            
    def predict(self,T,Z,X):
        
        prediction = getFactorProjection(self.projectionType,self.projection,T,Z)
        
        if self.remainderRegressor is not None:
            prediction += self.remainderRegressor.predict(X)
        
        return prediction


###############################
# Regressor classes
###############################

# Regressor base class
class Regressor(object):
    
    def __init__(self):
        return

    def fit(self,X,y): 
        return self

    def predict(self,X):
        raise Exception('Unhandled predict method')
            
    def getRisks(self,X):
        # manual bump and recompute
        dx = .0001
        fX = self.predict(X)
        fXDown = self.predict(X*(1-dx))
        fXUp = self.predict(X*(1+dx))
        
        deltas = (fXUp-fXDown) / (2.*X*dx)
        gammas = (fXUp-2.*fX+fXDown) / (X*dx)**2
        
        return deltas, gammas
    
# Regressor wrapper from a function
class FromFunctionRegressor(Regressor):
    
    def __init__(self,f=None):
        super().__init__()
        
        self.f = f
        
    def predict(self,X):
        return self.f(X).reshape((-1))


# Simple polynomial regressor
class PolynomialRegressor(Regressor):
    
    def __init__(self,alpha=0.0,d=2):
        super().__init__()
        
        self.alpha = alpha
        self.d = d

        # initialise the polynomial regressor
        self.polynomialTransformer = preprocessing.PolynomialFeatures(self.d,False,False)
        self.regressor = linear_model.Ridge(alpha=self.alpha,fit_intercept=True,normalize=True,solver='cholesky')

    def fit(self,X,y,sample_weight=None):        
        data = self.polynomialTransformer.fit_transform(X)
        self.regressor.fit(data,y,sample_weight)
        return self
            
    def predict(self,X):
        data = self.polynomialTransformer.fit_transform(X)
        return self.regressor.predict(data)


# Chebyshev regressor
class ChebyshevRegressor(Regressor):
    
    def __init__(self,d=2):
        super().__init__()
        
        self.d = d
        self.chebyshevRegressor = None

    def fit(self,X,y):
        
        if X.shape[1] > 1:
            raise Exception('Chebyshev multivariate not available yet')
        
        # save borders as we will extrapolate flat
        self.minAbscissa = np.min(X)
        self.maxAbscissa = np.max(X)
        
        self.chebyshevRegressor = np.polynomial.Chebyshev.fit(X.reshape((-1)),y.reshape((-1)),self.d)
        
        self.minAbscissaPrediction = self.chebyshevRegressor(self.minAbscissa)
        self.maxAbscissaPrediction = self.chebyshevRegressor(self.maxAbscissa)

        return self
            
    def predict(self,X):
                
        if X.shape[1] > 1:
            raise Exception('Chebyshev multivariate not available yet')
        
        newX = X.reshape((-1))
        
#        predictions = self.minAbscissaPrediction * (newX<self.minAbscissa) \
#            + self.chebyshevRegressor(newX) * (newX>=self.minAbscissa) * (newX<=self.maxAbscissa) \
#            + self.maxAbscissaPrediction * (newX>self.maxAbscissa)
        
        predictions = self.chebyshevRegressor(newX)
            
        return predictions


# Regressor inside a region with cap and floor forecast
class CapFloorRegionRegressor(Regressor):
    
    def resetRegion(self):

        # floor and cap of the payoff
        self.hasFloor = False
        self.floor = None
        self.hasCap = False
        self.cap = None
        
        # check if payoff is constant
        self.isCt = False
        self.emptyRegressor = False
        
        self.floorRegionIdx = None
        self.inBetweenRegionIdx = None
        self.capRegionIdx = None

    
    def __init__(self,classifierName='l',classifierParams={},regressorName='p',regressorParams={}):
        super().__init__()
        
        self.classifierName = classifierName
        self.classifierParams = classifierParams
        self.regressorName = regressorName
        self.regressorParams = regressorParams
        self.resetRegion()

        # also initialise the logistic regressor that will find regions projection
        self.clf = getClassifier(self.classifierName,self.classifierParams)
        
        # initialise the polynomial regressor
        self.regressor = getRegressor(self.regressorName,self.regressorParams)


    def transform(self,X):
        
        transformedX = X
#        for shift, factor in [(0.0,1.0)]:
#        for shift, factor in [(0.0,0.75),(0.0,1.0),(0.0,1.5)]:
#        for shift, factor in [(0.0,0.9),(0.0,0.95),(0.0,1.0),(0.0,1.05),(0.0,1.1)]:
#            transformedX = np.append(transformedX,callLike(X,shift,factor),axis=1)
        
        return transformedX
        
    
    def fit(self,X,y):
        
        # check payoff floor / cap
        self.resetRegion()
        self.hasFloor, self.floor, self.hasCap, self.cap = checkPayoffCapFloor(y)
        self.isCt = self.hasFloor and self.hasCap and isclose(self.floor,self.cap)
        if self.isCt:
            return self

        transformedX = self.transform(X)
        if self.hasFloor or self.hasCap:
            
            # label payoffs according to the region found (floor=0, inBetween=1, cap=2)
            yLabels = np.empty_like(y)
            yLabels[y==self.floor] = 0
            inBetween = (1-(y==self.cap))*(1-(y==self.floor))==1
            yLabels[inBetween] = 1
            yLabels[y==self.cap] = 2
            
            #- check for labels idx when forecasting probas
            nbInBetween = np.sum(inBetween)
            if self.hasFloor:
                self.floorRegionIdx = 0
                if nbInBetween !=0:
                    self.inBetweenRegionIdx = 1
                    self.capRegionIdx = 2 if self.hasCap else None
                else:
                    self.inBetweenRegionIdx = None
                    self.capRegionIdx = 1 if self.hasCap else None
            else:
                self.floorRegionIdx = None
                if nbInBetween !=0:
                    self.inBetweenRegionIdx = 0
                    self.capRegionIdx = 1
                else:
                    self.inBetweenRegionIdx = None
                    self.capRegionIdx = 0
                            

            # train the classifier on these labels
            self.clf.fit(transformedX,yLabels)

            # train the polynomial regressor only in between floor and cap
            if nbInBetween != 0:
                self.regressor.fit(transformedX[yLabels==1,:],y[yLabels==1])
            else:
                # case where we have only caps and floors, nothing in between
                self.emptyRegressor = True
        
        else:        
            self.regressor.fit(transformedX,y)
            
        return self
            
    def predict(self,X):
        
        if self.isCt:
            return self.floor * np.ones((X.shape[0]))
        
        transformedX = self.transform(X)
        
        if self.hasFloor or self.hasCap:
            
            regionsProba = self.clf.predict_proba(transformedX)
            forecasts = None
            if self.emptyRegressor:
                forecasts = np.zeros(X.shape[0])
            else:
                forecasts = self.regressor.predict(transformedX) * regionsProba[:,self.inBetweenRegionIdx]
            
            if self.hasFloor:
                forecasts += self.floor * regionsProba[:,self.floorRegionIdx]
            if self.hasCap:
                forecasts += self.cap * regionsProba[:,self.capRegionIdx]
    
            return forecasts
        
        else:
        
            return self.regressor.predict(transformedX)


# Data points are clustered using kMeans algorithm and a specific regressor is attached to each cluster
class ClusteredRegressor(Regressor):
     
    def __init__(self,nbClusters=5,clusteringFraction=1.0,clusteringFractionRandomSelect=False,smoothing=False,smoothingNbNeighbors=2,smoothingGamma=None,regressorName='p',regressorParams={}):
        super().__init__()
        
        self.nbClusters = nbClusters
        self.clusteringFraction = clusteringFraction
        self.clusteringFractionRandomSelect = clusteringFractionRandomSelect

        self.smoothing = smoothing
        self.smoothingNbNeighbors = smoothingNbNeighbors
        self.smoothingGamma = smoothingGamma

        self.regressorName = regressorName
        self.regressorParams = regressorParams
        
        self.kMeans = None
        
        self.fitLabels = None
        self.regressors = []
        self.caps = []
        self.floors = []


    def fitSet(self):
        # returns the centroids and labels of the dataset used that was fit
        return self.kMeans.cluster_centers_, self.fitLabels
        
    
    def fit(self,X,y):
        
        self.regressors = []
        
        # cluster points first
        # select a fraction of the incoming dataset for clustering only
        nbPointsForClustering = ceil(self.clusteringFraction * X.shape[0])
        clusteringDataStart = 0
        if self.clusteringFractionRandomSelect:
            clusteringDataStart = np.random.randint(0,X.shape[0]-nbPointsForClustering)

        # create and fit the clustering algo
        clusteringData = X[clusteringDataStart:clusteringDataStart+nbPointsForClustering]
        self.kMeans = cluster.KMeans(init=clusteringData[:self.nbClusters], n_clusters=self.nbClusters, n_init=1)
        self.kMeans.fit(clusteringData)

        # now get labels for all the points in the training dataset
        self.fitLabels = self.kMeans.predict(X)

        # then regress inside each cluster
        centroids = self.kMeans.cluster_centers_
        for iCls in range(self.nbClusters):

            clusterRegressor = getRegressor(self.regressorName,self.regressorParams)

            # check for caps or floors in the payoff
            clusterFloor = None
            clusterCap = None
            clusterHasFloor, clusterFloor, clusterHasCap, clusterCap = checkPayoffCapFloor(y[self.fitLabels==iCls])
            self.floors.append(clusterFloor)
            self.caps.append(clusterCap)
                
            # reduce data to that cluster
            dataCls = X[self.fitLabels==iCls]
            yCls = y[self.fitLabels==iCls]
        
            # check if there is actually sthg to regress inside that cluster
            if dataCls.shape[0] != 0:
                clusterRegressor.fit(dataCls,yCls)
                
            self.regressors.append(clusterRegressor)

        # check if we should perform smoothing using a kernel
        if self.smoothing:
            
            smoothedRegressors = []
            
            # get prediction using the regressors that have just been fit
            smoothingY = self.predict(X)

            for iCls in range(self.nbClusters):
    
                clusterRegressor = getRegressor(self.regressorName,self.regressorParams)

                # create a gaussian kernel and compute weigths that will be applied to each cluster point
                clustersWeights = metrics.pairwise.rbf_kernel(centroids,centroids[iCls].reshape((1,-1)),self.smoothingGamma).reshape((-1))
            
                # just keep k closest clusters
                selectedClusters = np.argsort(clustersWeights)[::-1][:self.smoothingNbNeighbors+1]
                selectedPoints = np.isin(self.fitLabels,selectedClusters)
                
                # reduce data for that cluster
                dataCls = X[selectedPoints]
                yCls = smoothingY[selectedPoints]
            
                dataWeights = clustersWeights[self.fitLabels[selectedPoints]]
            
                # run regression
                clusterRegressor.fit(dataCls,yCls,dataWeights)
                
                smoothedRegressors.append(clusterRegressor)

            self.regressors = smoothedRegressors
    
        return self
            
    
    def predict(self,X):
        
        # first find clusters the points belong to
        labels = self.kMeans.predict(X)
        
        # run predictions for each cluster
        predictions = np.zeros(X.shape[0])
        for iCls in range(self.nbClusters):
            if np.sum(labels==iCls) != 0:
                dataCls = X[labels==iCls]
                predictions[labels==iCls] = self.regressors[iCls].predict(dataCls)
#                if self.caps[iCls] is not None:
#                    predictions[labels==iCls] = np.minimum(self.caps[iCls],predictions[labels==iCls])
#                if self.floors[iCls] is not None:
#                    predictions[labels==iCls] = np.maximum(self.floors[iCls],predictions[labels==iCls])
                    
        return predictions
    

    def getRisks(self,X):
        
        # first find clusters the points belong to
        labels = self.kMeans.predict(X)
        
        # get risks for each cluster
        deltas = np.zeros(X.shape[0])
        gammas = np.zeros(X.shape[0])
        for iCls in range(self.nbClusters):
            if np.sum(labels==iCls) != 0:
                dataCls = X[labels==iCls]
                deltas[labels==iCls], gammas[labels==iCls] = self.regressors[iCls].getRisks(dataCls)
        
        return deltas, gammas


# Averaging regression on several sub-training sets
class AveragedRegressor(Regressor):
          
    def __init__(self,subRegressorsSetup):
        super().__init__()
        
        self.subRegressorsSetup = subRegressorsSetup
        self.nbSubRegressors = len(self.subRegressorsSetup)
    
    def fit(self,X,y):
        
        self.weights = []
        self.regressors = []
        
        # train each sub-regressor
        self.scaling = .0
        for subRegSetup in self.subRegressorsSetup:

            weight = subRegSetup.get('regressorWeight',1./self.nbSubRegressors)
            self.weights.append(weight)
            self.scaling += weight
            
            subRegressor = getRegressor(subRegSetup.get('regressorName','p'),subRegSetup.get('regressorParams',{}))
            subRegressor.fit(X,y)

            self.regressors.append(subRegressor)
            
        return self
            
  
    def predict(self,X):
        
        prediction = np.zeros(X.shape[0])
        for subRegIdx, subReg in enumerate(self.regressors):
            prediction += self.weights[subRegIdx] * subReg.predict(X)
            
        return prediction / self.scaling


    def getRisks(self,X):
        
        deltas = np.zeros(X.shape[0])
        gammas = np.zeros(X.shape[0])
        for subRegIdx, subReg in enumerate(self.regressors):
            regDeltas, regGammas = subReg.getRisks(X)
            deltas += self.weights[subRegIdx] * regDeltas
            gammas += self.weights[subRegIdx] * regGammas
            
        return deltas / self.scaling, gammas / self.scaling


# Factory
# Add test regressors here
def getRegressor( regressorName, regressorParams):
    
    if regressorName == 'p':
        return PolynomialRegressor(**regressorParams)
    elif regressorName == 'ch':
        return ChebyshevRegressor(**regressorParams)
    elif regressorName == 'c':
        return ClusteredRegressor(**regressorParams)
    elif regressorName == 'a':
        return AveragedRegressor(**regressorParams)
    elif regressorName == 'rf':
        return ensemble.RandomForestRegressor(**regressorParams) #(max_depth=5)
    elif regressorName == 'cfr':
        return CapFloorRegionRegressor(**regressorParams)
    elif regressorName == 'svr':
        return svm.SVR(**regressorParams)
    elif regressorName == 'kr':
        return kernel_ridge.KernelRidge(**regressorParams)
    else:
        raise Exception('Unhandled regression method: ' + regressorName)


def getClassifier( classifierName, classifierParams):
    if classifierName == 'l':
        return LogisticClassifier(**classifierParams)
    else:
        raise Exception('Unhandled classification method: ' + classifierName)

regressorNameLabelMapping = {
    'p': 'Polynomial Regression',
    'ch': 'Chebyshev Regression',
    'c': 'Clustered Regression',
    'a': 'Averaged Regression',
    'rf': 'Random Forest Regression',
    'cfr': 'Cap Floor Region Regression',
    'svr': 'Support Vector Regression',
    'kr': 'Kernel Ridge Regression'
}

def getRegressorLabel(testRegressorName):    
    if testRegressorName in regressorNameLabelMapping:
        return regressorNameLabelMapping[testRegressorName]
    else:
        return testRegressorName
