#%%
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(752)
from Regression import *
from Exposures import *

#############################
# Setup
#############################

# Model setup
S = 1.0
model = 'ln'
sigma = 0.2

# Simulation setup
T = 5.0
step = .5
nbPreSimSamples = 100000
nbScenariosSamples = 100000

def Digit(Call,strike,overhedge):      # digit with overhedges
    return ( Call(strike-.5*overhedge) - Call(strike+.5*overhedge) ) / overhedge

def Range(Call,strikeDown,strikeUp,overhedge):      # range with overhedges
    return Digit(Call,strikeDown,overhedge) - Digit(Call,strikeUp,overhedge)

# Write payoff here
def payoffFn(Fwd,Call,Put):
    
#    return Call(1.)
#     return Put(1.)
#     return Call(.9) - Call(1.1)
#     return Fwd
#     return Call(1.) + Put(1.)
    return Range(Call,0.8,1.2,0.1)
#    return Range(Call,1.2,1.6,0.1)
#    return Range(Call,0.4,0.8,0.1)
#    return Range(Call,0.7,0.9,0.1) - Range(Call,1.1,1.3,0.1)


# Regression setup
projectedRegression = True # project on a set of calls on udl factor and regress only the remainder
projectionNbPayoffs = 11
projectionPruning = 1.
projectionRegressRemainder = True
projectionType = 'calls' # project on 'calls' or 'moments'
projectionDegree = 3

chainedRegression = True
densifiedChainedRegression = False
densificationNbSamples = 10

testRegressorName = 'c' # 'c' for clustered, 'p' for polynomial, 'rf' for random forest...
testRegressorParams = {
    # polynomial regressor
    'p': {
        'alpha': 1e-8,
        'd': 2
    },
    # chebyshev regressor
    'ch': {
        'd': 10
    },
    # clustered regressor
    'c': {
        'nbClusters': 10,
        'clusteringFraction': .1,
        'smoothing': False,
        'smoothingNbNeighbors': 1,
        'smoothingGamma': 10.,
        'regressorName': 'p',
        'regressorParams': {
            'alpha': 1e-15,
            'd': 3
        }
    },
    # random forest regressor
    'rf': {
        'max_depth': 5
    },
    # support vector regressor
    'svr': {
        'C': 1.0,
        'kernel': 'rbf'
    },
    # kernel ridge regressor
    'kr': {
        'alpha': 1e-8,
        'kernel': 'rbf'
    },
    # averaged regressor
    'a': {
        'subRegressorsSetup': [
            {
                'regressorWeight': 0.75,
                'regressorName': 'c',
                'regressorParams': {
                    'nbClusters': 10,
                    'clusteringFraction': .1,
                    'regressorName': 'p',
                    'regressorParams': {
                        'alpha': 1e-8,
                        'd': 3
                    }
                }
            },
            {
                'regressorWeight': 0.25,
                'regressorName': 'c',
                'regressorParams': {
                    'nbClusters': 25,
                    'clusteringFraction': .1,
                    'regressorName': 'p',
                    'regressorParams': {
                        'alpha': 1e-8,
                        'd': 3
                    }
                }
            }
        ]
    }
}

testRegressOnFormula = False # regress directly on the formula provided eg to validate the functional fit of the regressor 

# Plot setup

# During regression
plotPayoffDistribution = False

plotBinnedStatistics = False

plotClusterLabels = False

plotRegressions = False
plotFormulaPayoffWithRegressions = True

# During exposures monitoring
plotForecasts = True

plotForecastsVsFormulaRegression = False

# Exposures analysis setup
useControlVariate = False
thresholdsCoverage = 1.0 # coverage of formula span
nbThresholdsSteps = 100 # nb of thresholds steps to split forumla span
thresholdsEpeRelativeDiffs = False # Compute relative (vs absolute) differences vs EPE with formula
addDiffsWithCoupons = False # Also compute diffs for the method using 'Coupons' payoff and regression only in the indicator


#############################
# Regression step
#############################

testRegressors = runRegression(payoffFn,S,T,step,model,sigma,nbPreSimSamples,projectedRegression,projectionNbPayoffs,projectionPruning,projectionRegressRemainder,projectionType,projectionDegree,chainedRegression,densifiedChainedRegression,densificationNbSamples,testRegressorName,testRegressorParams.get(testRegressorName,{}),testRegressOnFormula,plotPayoffDistribution,plotBinnedStatistics,plotClusterLabels,plotRegressions,plotFormulaPayoffWithRegressions)

#############################
# Exposures step
#############################

exposures = runExposuresAnalysis( payoffFn, S, T, step, model, sigma, nbScenariosSamples, useControlVariate, testRegressors, testRegressorName, thresholdsCoverage, nbThresholdsSteps, plotForecasts, plotForecastsVsFormulaRegression)

#############################
# Exposures at Regression Dates
#############################

#plotAllExposures( exposures, addDiffsWithCoupons, testRegressorName)

#############################
# Exposures with Thresholds at Regression Dates
#############################

plotThresholdExposures(exposures, thresholdsEpeRelativeDiffs, addDiffsWithCoupons, testRegressorName)

#############################
# Exposures with Thresholds at a specific Date
#############################

atDate = 4
#plotThresholdExposuresAtDate( exposures, atDate, addDiffsWithCoupons, testRegressorName)
