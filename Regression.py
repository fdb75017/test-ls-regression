# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np, scipy as sp
import time
from functools import partial

from Simulation import getSimulations, generateDenseSetOnStep
from Payoffs import evalPayoff
from Regressors import ProjectionRegressor, FromFunctionRegressor, getRegressor, getRegressorLabel, getFactorProjection


def runRegression(payoffFn, S, T, step, model, sigma,
                  nbPreSimSamples,
                  projectedRegression = False, projectionNbPayoffs = None, projectionPruning = 1., projectionRegressRemainder = True, projectionType = 'calls', projectionDegree = 2,
                  chainedRegression = False, densifiedChainedRegression = False, densificationNbSamples = None,
                  testRegressorName = 'p', testRegressorParams = {},
                  testRegressOnFormula = False,
                  plotPayoffDistribution = False, plotBinnedStatistics = False, plotClusterLabels = False, plotRegressions = False, plotFormulaPayoffWithRegressions = False):
    
    # Run pre-simulation
    dates, simulations, factors = getSimulations(S,T,step,model,sigma,nbPreSimSamples,True)

    # Eval payoff at maturity
    def payoffAtMaturityFn(S):
        return evalPayoff(payoffFn,0.,S)

    payoff = payoffAtMaturityFn(simulations[:,-1])

    if plotPayoffDistribution:
        
        # Plot empirical distribution of the payoff at maturity
        bin_count_payoff, bin_edges_payoff, binnumber_payoff = sp.stats.binned_statistic(payoff, np.zeros(payoff.shape), 'count', bins=100)

        plt.figure()
        plt.title('Empirical payoff distribution at maturity')
        plt.plot(bin_edges_payoff[:-1],bin_count_payoff/nbPreSimSamples,'x')
        plt.show()

    if testRegressOnFormula:
        print('Warning: regressing directly on Formula')

    # Build regressors for each date
    testRegressors = []
    totalRegressionTime = .0
    testReg = prevTestReg = prevRegressorData = None
    testRegressorTarget = payoff
    
    # check if we want to decompose the payoff at maturity on a set of calls on the udl factor
    factorProjection = None
    if projectedRegression:
        
        regressionStartTime = time.time()
        
        if projectionType == 'calls':
        
            # build the set of factor calls payoffs
            factorStd = np.std(factors[:,-1])
            factorCallsStrikes = None
            if projectionNbPayoffs==1:
                factorCallsStrikes = np.array([.0]).reshape((1,1))
            else:            
                factorCallsStep = 2. * projectionPruning * factorStd / (projectionNbPayoffs-1)
                factorCallsStrikes = (- projectionPruning * factorStd + factorCallsStep * np.arange(projectionNbPayoffs)).reshape((1,projectionNbPayoffs))
            
    #        print('Factor stdev: '+str(factorStd))
    #        print('Factor calls strikes: '+str(factorCallsStrikes))
            
            factorCalls = np.maximum(factors[:,-1].reshape((-1,1))-factorCallsStrikes,0.)
            
            # add factor fwd to be able to regress puts easily...
            factorProjectionData = np.hstack((factors[:,-1].reshape((-1,1)),factorCalls))
            
            # regress the real payoff on it
            factorCallsRegressor = getRegressor( 'p', {'d':1,'alpha': 1e-8})
            factorCallsRegressor.fit(factorProjectionData,payoff)
            
            # log the factor calls functions, the decomposition and regress on remainders
            factorCallsRegressionRemainders = payoff - factorCallsRegressor.predict(factorProjectionData)
            
    #        print('Payoff stdev: '+str(np.std(payoff)))
    #        print('Remainder stdev: '+str(np.std(factorCallsRegressionRemainders)))
            
            factorProjection = {
                'strikes': factorCallsStrikes,
                'decomposition': factorCallsRegressor
            }
            
            # reset the regression target
            testRegressorTarget = factorCallsRegressionRemainders

        else: 
            
            # project on factor moments
            if projectionDegree<=0 or projectionDegree>6:
                raise Exception('Factor moments projection degree must be between 1 and 6')
            
            factorProjectionData = np.power(factors[:,-1].reshape((-1,1)),1+np.arange(projectionDegree))
            
            factorMomentsRegressor = getRegressor( 'p', {'d':1,'alpha': 1e-8})
            factorMomentsRegressor.fit(factorProjectionData,payoff)
            
            # log the factor moments functions, the decomposition and regress on remainders
            factorMomentsRegressionRemainders = payoff - factorMomentsRegressor.predict(factorProjectionData)
            
    #        print('Payoff stdev: '+str(np.std(payoff)))
    #        print('Remainder stdev: '+str(np.std(factorMomentsRegressionRemainders)))
            
            factorProjection = {
                'degree': projectionDegree,
                'decomposition': factorMomentsRegressor
            }
            
            # reset the regression target
            testRegressorTarget = factorMomentsRegressionRemainders


        regressionEndTime = time.time()
        totalRegressionTime += (regressionEndTime-regressionStartTime)

    
    # check whether we actually need to regress (either payoff itself or remainder after projection)
    doRegress = (not projectedRegression) or  projectionRegressRemainder
    
    for k, date in enumerate(dates[:0:-1]):

        # Regress at Tk
        regressorData = simulations[:,-1-k].reshape((nbPreSimSamples,1))
        Z = factors[:,-1-k].reshape((nbPreSimSamples,1))

        # exact payoff evaluated on MC paths
        payoffFormula = evalPayoff(payoffFn,dates[-1]-date,regressorData,model,sigma).reshape((-1)) \
            - getFactorProjection(projectionType,factorProjection,dates[-1]-date,Z).reshape((-1))

        # test regressor
        regressionStartTime = time.time()
        
        if k==0:
            # no need to run a regression at payoff date, just use the payoff function
            testReg = ProjectionRegressor( projectionType, None, FromFunctionRegressor(payoffAtMaturityFn) )
        else: 
            
            testReg = None
            if doRegress:
                # create new test reg
                testReg = getRegressor(testRegressorName,testRegressorParams)
            
                # perform regression
                if testRegressOnFormula:
                    # test regress on formula directly provided (eg if able to identify the payoff...)
                    # useful to check the functional fit of the regressor
                    testReg.fit(regressorData,payoffFormula.reshape((-1)))
                else:
                    # get the target to regress on
                    # by default direclty regress on corresponding payoff realizations, a-la LS
                    if chainedRegression and k!=1:
    
                        # here we regress on the previous regression values
                        if densifiedChainedRegression:
    
                            # densify and call the previous regressor then get averages on densification sets
                            nextStep = dates[-k] - date
                            densifiedRegressorData = generateDenseSetOnStep(regressorData,nextStep,model,sigma,densificationNbSamples,True)
                
    #                        densifiedRegressorData = np.repeat(prevRegressorData,densificationNbSamples,axis=0)
                            
                            # get the corresponding 'next' regression realizations
                            densifiedForecastData = prevTestReg.remainderRegressor.predict(densifiedRegressorData)
                            
                            # reshape and average it
                            testRegressorTarget = np.mean(densifiedForecastData.reshape(-1, densificationNbSamples), axis=1)
                            
                        else:
                            # just get the corresponding 'next' regression realizations
                            testRegressorTarget = prevTestReg.remainderRegressor.predict(prevRegressorData)
                    
                    # do the actual regression for this date
                    testReg.fit(regressorData,testRegressorTarget)
                
            # wrap the regressor with the factor calls projection
            testReg = ProjectionRegressor( projectionType, factorProjection, testReg )
            
        
        testRegressors.insert(0,testReg)

        regressionEndTime = time.time()
        totalRegressionTime += (regressionEndTime-regressionStartTime)

        plotAbscissa = np.log(regressorData) if model == 'ln' else regressorData
                
        if doRegress and plotBinnedStatistics:
            
            # Plot local averages and variance
            fig, ax1 = plt.subplots()
            plt.title('Maturity T='+str(T)+'. Binned stats at Tobs=' + str(date))

            bin_means, bin_edges, binnumber = sp.stats.binned_statistic(regressorData.reshape((nbPreSimSamples)), testRegressorTarget.reshape((nbPreSimSamples)), 'mean', bins=100)
            
            bin_counts, bin_edges_counts, binnumber = sp.stats.binned_statistic(regressorData.reshape((nbPreSimSamples)), testRegressorTarget.reshape((nbPreSimSamples)), 'count', bins=100)
            
            averagesMidPoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            ax1.plot(plotAbscissa, testRegressorTarget, 'o', label="Payoff")
            plotAveragesAbscissa = np.log(averagesMidPoints) if model == 'ln' else averagesMidPoints
            ax1.plot(plotAveragesAbscissa, bin_means, 'o', label="Local Averages")
            
            ax2 = ax1.twinx()
            ax2.plot(plotAveragesAbscissa, bin_counts/nbPreSimSamples, 'x', label="Empirical distribution", color="green")

            ax1.legend(loc=0)
            ax2.legend(loc=1)
            plt.show()

        if doRegress and plotClusterLabels and testRegressorName == 'c' and k != 0:
            
            # Plot clusters labels
            
            # Reorder them first
            centroids, labels, gammas = testReg.remainderRegressor.fitSet()
            sortedLabelsIndices = np.argsort(centroids,axis=0)
            newClusters = np.empty((nbPreSimSamples,1))
            for labelIdx in range(testReg.remainderRegressor.nbClusters):
                newClusters[labels==sortedLabelsIndices[labelIdx],0] = labelIdx
            
            fig, ax1 = plt.subplots()
            plt.title('Maturity T='+str(T)+'. Cluster Labels at Tobs=' + str(date))
            ax1.plot(plotAbscissa, testRegressorTarget, 'o', label="Payoff")
            ax2 = ax1.twinx()
            ax2.plot(plotAbscissa, newClusters, 'x', label='Clusters', color="orange")
            
            ax1.legend(loc=0)
            ax2.legend(loc=1)
            plt.show()
            
            # Also plot % of points found in each cluster
            sortedCentroids = centroids[sortedLabelsIndices.reshape((testReg.remainderRegressor.nbClusters))]
            plotCentroids = np.log(sortedCentroids) if model == 'ln' else sortedCentroids
            newClustersUniqueElements, newClustersCountElements = np.unique(newClusters, return_counts=True)
            fig, ax1 = plt.subplots()
            plt.title('Maturity T='+str(T)+'. Cluster % of points at Tobs=' + str(date))
            ax1.plot(plotCentroids,newClustersCountElements/nbPreSimSamples, 'o')
            #print(newClustersCountElements/nbPreSimSamples)
            plt.show()

            
        if doRegress and plotRegressions:

            # Plot regressions
            plt.figure()
            plt.title('Maturity T='+str(T)+'. Regression at Tobs=' + str(date)+'\nFit on: '+('Formula' if testRegressOnFormula else 'Payoff'))

            plt.plot(plotAbscissa, testRegressorTarget, 'o', label="Payoff")
            if k==0:
                plt.plot(plotAbscissa, testRegressorTarget, 'o', label=getRegressorLabel(testRegressorName))
            else:
                plt.plot(plotAbscissa, testReg.remainderRegressor.predict(regressorData), 'o', label=getRegressorLabel(testRegressorName))

            if plotFormulaPayoffWithRegressions:
                
                plt.plot(plotAbscissa, payoffFormula, 'o', label="Formula")
                
                # plot chebyshev regression on formula
#                formulaRegressor = getRegressor('ch',{'d':50})
#                formulaRegressor.fit(regressorData,payoffFormula)
#                plt.plot(plotAbscissa, formulaRegressor.predict(regressorData), 'o', label='Chebyshev on formula')

            # plot local averages
            bin_means, bin_edges, binnumber = sp.stats.binned_statistic(regressorData.reshape((nbPreSimSamples)), testRegressorTarget.reshape((nbPreSimSamples)), 'mean', bins=50)
            averagesMidPoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            plotAveragesAbscissa = np.log(averagesMidPoints) if model == 'ln' else averagesMidPoints
#            plt.plot(plotAveragesAbscissa, bin_means, 'o', label="Local Averages")
            plt.legend()
            plt.show()
            
            # Also plot 'risks' per cluster
#            if testRegressorName == 'c' and k != 0:
#                centroids, labels, gammas = testReg.remainderRegressor.fitSet()
#                plotCentroids = np.log(centroids) if model == 'ln' else centroids
#                fig, ax1 = plt.subplots()
#                plt.title('Maturity T='+str(T)+'. Cluster gamma at Tobs=' + str(date))
#                ax1.plot(plotCentroids,gammas, 'o')
#                #print(newClustersCountElements/nbPreSimSamples)
#                plt.show()

        prevTestReg = testReg
        prevRegressorData = regressorData


    print("\nRegression time: " + "{:.2f}".format(totalRegressionTime) + " seconds")
    
    return testRegressors
		
