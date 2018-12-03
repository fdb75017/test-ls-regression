from matplotlib import pyplot as plt, cm as cm
import numpy as np, scipy as sp
from sklearn import linear_model
from math import sqrt
import time

# -*- coding: utf-8 -*-

from Simulation import getSimulations
from Payoffs import evalPayoff
from Regression import getRegressorLabel

def covar(x,y):
    resX = x.reshape((-1,1))
    resY = y.reshape((-1,1))
    return np.average(resX*resY)-np.average(resX)*np.average(resY)

# MTFs have shape (nbSamples,1)
# thresholds has shape (1,nbThresholds)
def getExposures(indicMTFs,payoffMTFs,thresholds,useControlVariate=False,pv=None):

    eePayoff = payoffMTFs
    epePayoff = payoffMTFs*(indicMTFs>0.0)
    enePayoff = payoffMTFs*(indicMTFs<0.0)
    stDevMcFactor = 1. / sqrt(payoffMTFs.shape[0])
    
    # EPE with thresholds
    payoffsTh = payoffMTFs - thresholds
    indicsTh = indicMTFs - thresholds
    epeThresholdsPayoff = payoffsTh*(indicsTh>0.0)
    
    # adjustments with control variates
    if useControlVariate and pv is not None:
        eeVarScaling = np.var(eePayoff)
        epePayoff -= covar(epePayoff,eePayoff) / eeVarScaling * ( eePayoff - pv )
        enePayoff -= covar(enePayoff,eePayoff) / eeVarScaling * ( eePayoff - pv )
        for thIdx in range(thresholds.shape[0]):
            epeThresholdsPayoff[:,thIdx] -= covar(epeThresholdsPayoff[:,thIdx],eePayoff) / eeVarScaling * ( eePayoff.reshape((-1)) - pv )
        
    return {
        'ee': np.average(eePayoff),
        'eeStDev': np.std(eePayoff)*stDevMcFactor,
        'epe': np.average(epePayoff),
        'epeStDev': np.std(epePayoff)*stDevMcFactor,
        'ene': np.average(enePayoff),
        'eneStDev': np.std(enePayoff)*stDevMcFactor,
        'epeThresholds': np.average(epeThresholdsPayoff,axis=0),
        'epeThresholdsStDev': np.std(epeThresholdsPayoff,axis=0)*stDevMcFactor
    }


def runExposuresAnalysis(payoffFn,S,T,step,model,sigma,
                  nbScenariosSamples, useControlVariate,
                  testRegressors, testRegressorName,
                  thresholdsCoverage, nbThresholdsSteps,
                  plotForecasts = False,
                  plotForecastsVsFormulaRegression = False):
    
    # Run simulation
    dates, simulations, factors = getSimulations(S,T,step,model,sigma,nbScenariosSamples,False)

    # Eval payoff at maturity
    payoff = evalPayoff(payoffFn,.0,simulations[:,-1],model,sigma).reshape((-1,1))
    pvMC = np.average(payoff)
    pvFormula = evalPayoff(payoffFn,T,S,model,sigma)

    # Compute exposures for each date
    exposuresForecastTime = 0.0
    thresholds = []
#    formulaHistogram = []
    exposuresExactFormula = []
    exposuresTestReg = []
    exposuresTestRegWithCoupons = []
    for k, date in enumerate(dates[1:]):
        
        # Forecast at Tk
        regressorData = simulations[:,k+1].reshape((-1,1))
        Z = factors[:,k+1].reshape((-1,1))
        
        # Exact payoff evaluated on MC paths
        payoffFormula = evalPayoff(payoffFn,dates[-1]-date,regressorData,model,sigma)

        # Regression results
        exposuresStartTime = time.time()
        testForecast = testRegressors[k].predict(dates[-1]-date,Z,regressorData).reshape((-1,1))
        exposuresEndTime = time.time()
        exposuresForecastTime += (exposuresEndTime-exposuresStartTime)
    
        if plotForecasts:

            # Plot forecasts
            plt.figure()
            plt.title('Maturity T='+str(T)+'. Forecasts at Tobs=' + str(date))

            plotAbscissa = np.log(regressorData) if model == 'ln' else regressorData
            plt.plot(plotAbscissa, payoffFormula, 'o', label="Formula")
            plt.plot(plotAbscissa, testForecast, 'o', label=getRegressorLabel(testRegressorName))

            plt.legend()
            plt.show()

        if plotForecastsVsFormulaRegression:

            # Plot forecasts
            forecastsToPlot = testForecast
            
            plt.figure()
            plt.title('Maturity T='+str(T)+'. Forecasts vs Formula at Tobs=' + str(date))
            plt.plot(payoffFormula, payoffFormula, 'r--')
            plt.plot(payoffFormula, forecastsToPlot, 'o')
            # add trendline
            trendRegress = linear_model.Ridge(alpha=0.,fit_intercept=False,normalize=False,solver='cholesky')
            trendRegress.fit(payoffFormula,forecastsToPlot)
            plt.plot(payoffFormula,trendRegress.predict(payoffFormula),"g--", label="y=%.6fx"%(trendRegress.coef_[0]))
          
            plt.legend()  
            plt.show()

        # Add EE, EPE, ENE
        
        # Compute Thresholds as a split of n equal steps of the coverage fraction of the formulae exposures span
        thresholdMin = max(1e-8,np.percentile(payoffFormula,50*(1.0-thresholdsCoverage)))
        thresholdMax = np.percentile(payoffFormula,50*(1.0+thresholdsCoverage))
        thresholdsSteps = (thresholdMax-thresholdMin) / nbThresholdsSteps

        thresholdsAtDate = np.arange(thresholdMin,thresholdMax+0.5*thresholdsSteps,thresholdsSteps)
        thresholds.append(thresholdsAtDate)

#        histFormula, binEdgesFormula = np.histogram(payoffFormula, thresholdsAtDate, density=True)
#        formulaHistogram.append(histFormula)
        
        exposuresExactFormula.append(getExposures(payoffFormula,payoffFormula,thresholdsAtDate,useControlVariate,pvMC))
        
        exposuresTestReg.append(getExposures(testForecast,testForecast,thresholdsAtDate,useControlVariate,pvMC))
        exposuresTestRegWithCoupons.append(getExposures(testForecast,payoff,thresholdsAtDate,useControlVariate,pvMC))


    print("\nForecast time: " + "{:.2f}".format(exposuresForecastTime) + " seconds")

    print("\nPV Formula: "+str(pvFormula))
    
    print("\nPV MC: "+str(pvMC))
    print("Diff MC: "+str(pvMC-pvFormula))
#    print("StDev MC: "+str(np.std(payoff)/sqrt(nbScenariosSamples)))
    
    print("\nPV Regress: "+str(exposuresTestReg[0]['ee']))
    print("Diff Regress: "+str(exposuresTestReg[0]['ee']-pvFormula))
#    print("StDev Regress: "+str(exposuresTestReg[0]['eeStDev']/sqrt(nbScenariosSamples)))
    
    return {
        'pvMC': pvMC,
        'pvFormula': pvFormula,
        'dates': dates,
        'thresholds': thresholds,
#        'formulaHistogram': formulaHistogram,
        'formula': exposuresExactFormula,
        'testReg': exposuresTestReg,
        'testRegWithCpns': exposuresTestRegWithCoupons
    }


def getAllExposures(exposures, exposuresName):
    return [x[exposuresName] for x in exposures]


def plotAllExposures( exposures, addDiffsWithCoupons, testRegressorName):

    # Plot exposures (EE, EPE, ENE) for both methodologies and compare with formula
    dates = exposures['dates']
    epeFormula = exposures['formula']
    epeTestReg = exposures['testReg']
    epeTestRegWithCpns = exposures['testRegWithCpns']
    
    fig, [ax1,ax2,ax3] = plt.subplots(1, 3, figsize=(20,6))
    # EE
    ax1.set_title('Expected Exposures')
    ax1.set_xlabel('Regression Date')
    ax1.set_ylabel('Exposure')
    ax1.plot(dates[1:], getAllExposures(epeFormula,'ee'), 'o', label="Formula")
    ax1.plot(dates[1:], getAllExposures(epeTestReg,'ee'), 'o', label=getRegressorLabel(testRegressorName))
    ax1.plot(dates[1:], exposures['pvFormula'] * np.ones((len(dates)-1)), '-', label="PV")
    ax1.plot(dates[1:], exposures['pvMC'] * np.ones((len(dates)-1)), '-', label="PV MC")
    ax1.legend()
    
    # EPE
    ax2.set_title('Expected Positive Exposures')
    ax2.set_xlabel('Regression Date')
    ax2.set_ylabel('Exposure')
    ax2.plot(dates[1:], getAllExposures(epeFormula,'epe'), 'o', label="Formula")
    ax2.plot(dates[1:], getAllExposures(epeTestReg,'epe'), 'o', label=getRegressorLabel(testRegressorName))
    ax2.legend()
    
    # ENE
    ax3.set_title('Expected Negative Exposures')
    ax3.set_xlabel('Regression Date')
    ax3.set_ylabel('Exposure')
    ax3.plot(dates[1:], getAllExposures(epeFormula,'ene'), 'o', label="Formula")
    ax3.plot(dates[1:], getAllExposures(epeTestReg,'ene'), 'o', label=getRegressorLabel(testRegressorName))
    ax3.legend()
    
    plt.show()

    if addDiffsWithCoupons:

        fig, [ax1,ax2,ax3] = plt.subplots(1, 3, figsize=(20,6))
        # EE
        ax1.set_title('Expected Exposures - Coupons mode')
        ax1.set_xlabel('Regression Date')
        ax1.set_ylabel('Exposure')
        ax1.plot(dates[1:], getAllExposures(epeFormula,'ee'), 'o', label="Formula")
        ax1.plot(dates[1:], getAllExposures(epeTestRegWithCpns,'ee'), 'o', label=getRegressorLabel(testRegressorName))
        ax1.legend()

        # EPE
        ax2.set_title('Expected Positive Exposures - Coupons mode')
        ax2.set_xlabel('Regression Date')
        ax2.set_ylabel('Exposure')
        ax2.plot(dates[1:], getAllExposures(epeFormula,'epe'), 'o', label="Formula")
        ax2.plot(dates[1:], getAllExposures(epeTestRegWithCpns,'epe'), 'o', label=getRegressorLabel(testRegressorName))
        ax2.legend()

        # ENE
        ax3.set_title('Expected Negative Exposures - Coupons mode')
        ax3.set_xlabel('Regression Date')
        ax3.set_ylabel('Exposure')
        ax3.plot(dates[1:], getAllExposures(epeFormula,'ene'), 'o', label="Formula")
        ax3.plot(dates[1:], getAllExposures(epeTestRegWithCpns,'ene'), 'o', label=getRegressorLabel(testRegressorName))
        ax3.legend()
    
        plt.show()


def plotThresholdExposures( exposures, thresholdsEpeRelativeDiffs, addDiffsWithCoupons, testRegressorName):
    
    dates = exposures['dates']
    thresholds = exposures['thresholds']
#    formulaHistogram = exposures['formulaHistogram']
    epeFormula = exposures['formula']
    epeTestReg = exposures['testReg']
    epeTestRegWithCpns = exposures['testRegWithCpns']
    
    nbDates = len(dates)-1
    nbThresholds = thresholds[0].shape[0]
    
    graphDates = np.repeat(dates[1:].reshape(nbDates,1),nbThresholds,axis=1)
    graphThresholds = np.empty((nbDates,nbThresholds))
#    graphFormulaHistogram = np.empty((nbDates,nbThresholds-1))
#    graphFormulaHistogramThresholds = np.empty((nbDates,nbThresholds-1))
    thresholdsEpeTest = np.empty((nbDates,nbThresholds))
    if addDiffsWithCoupons:
        thresholdsEpeTestWithCpns = np.empty((nbDates,nbThresholds))
        
    eeFormula = np.empty((nbDates))
    for k, date in enumerate(dates[1:]):
        eeFormula[k] = epeFormula[k]['ee']
        graphThresholds[k,:] = thresholds[k]
#        graphFormulaHistogramThresholds[k,:] = 0.5*(thresholds[k][:-1]+thresholds[k][1:])
#        graphFormulaHistogram[k,:] = formulaHistogram[k]
        if thresholdsEpeRelativeDiffs:
            thresholdsEpeTest[k,:] = np.abs(epeTestReg[k]['epeThresholds']-epeFormula[k]['epeThresholds']) / epeFormula[k]['epeThresholds']
            if addDiffsWithCoupons:
                thresholdsEpeTestWithCpns[k,:] = np.abs(epeTestRegWithCpns[k]['epeThresholds']-epeFormula[k]['epeThresholds']) / epeFormula[k]['epeThresholds']                
        else:
            thresholdsEpeTest[k,:] = np.abs(epeTestReg[k]['epeThresholds']-epeFormula[k]['epeThresholds'])
            if addDiffsWithCoupons:
                thresholdsEpeTestWithCpns[k,:] = np.abs(epeTestRegWithCpns[k]['epeThresholds']-epeFormula[k]['epeThresholds'])
    
    maxDiff = np.max(thresholdsEpeTest)

    # Plot formula histogram
#    plt.plot(figsize=(15,8))
#    cf = plt.contourf(graphDates[:,:-1], graphFormulaHistogramThresholds, graphFormulaHistogram, origin='lower', cmap=cm.Blues, vmin=0.0)
#    plt.colorbar(cf)
#    plt.title('Formula Mtf distribution')
#    plt.xlabel('Regression Date')
#    plt.ylabel('EPE Threshold')
#    plt.show()

    # Plot diffs
#    fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(15,8))
#    
#    cf1 = ax1.contourf(graphDates, graphThresholds, thresholdsEpeBase, origin='lower', cmap=cm.Blues, vmin=0.0)
#    ax1.plot(dates[1:],eeFormula,"x")
#    fig.colorbar(cf1, ax=ax1)
#    ax1.set_title('Base regression EPE diffs vs Formula')
#    ax1.set_xlabel('Regression Date')
#    ax1.set_ylabel('EPE Threshold')
#    
#    cf2 = ax2.contourf(graphDates, graphThresholds, thresholdsEpeTest, origin='lower', cmap=cm.Blues, vmin=0.0)#, vmax=maxDiff)
#    ax2.plot(dates[1:],eeFormula,"x")
#    fig.colorbar(cf2, ax=ax2)
#    ax2.set_title(getRegressorLabel(testRegressorName)+' EPE diffs vs Formula')
#    ax2.set_xlabel('Regression Date')
#    ax2.set_ylabel('EPE Threshold')
#
#    plt.show()
    
    print('\nMax EPE diff: '+str(maxDiff))
    
    plt.figure()
    plt.plot(dates[1:],eeFormula,"x")
    cf = plt.contourf(graphDates, graphThresholds, thresholdsEpeTest, origin='lower', cmap=cm.Blues, vmin=0.0)
    plt.colorbar(cf)
    plt.title(getRegressorLabel(testRegressorName)+' EPE diffs vs Formula')
    plt.xlabel('Regression Date')
    plt.ylabel('EPE Threshold')
    plt.show()


    if addDiffsWithCoupons:
    
        print('Max diff with cpns: '+str(np.max(thresholdsEpeTestWithCpns)))
    
        # Plot diffs vs using 'coupons' payoff method
        plt.figure()
        cf = plt.contourf(graphDates, graphThresholds, thresholdsEpeTestWithCpns, origin='lower', cmap=cm.Blues, vmin=0.0)
        plt.colorbar(cf)
        plt.title(getRegressorLabel(testRegressorName)+' EPE diffs vs Formula - Coupons mode')
        plt.xlabel('Regression Date')
        plt.ylabel('EPE Threshold')
        plt.show()
        
        plt.show()
    

def plotThresholdExposuresAtDate( exposures, atDate, addDiffsWithCoupons, testRegressorName):
    
    dates = exposures['dates']
    thresholds = exposures['thresholds']
    epeFormula = exposures['formula']
    epeTestReg = exposures['testReg']
    epeTestRegWithCpns = exposures['testRegWithCpns']

    dateIndex = np.where(dates[1:]==atDate)[0][0]
    thresholds = thresholds[dateIndex]
    thresholdsEpeFormula = epeFormula[dateIndex]['epeThresholds']
    thresholdsEpeTest = epeTestReg[dateIndex]['epeThresholds']
    thresholdsEpeTestWithCpns = epeTestRegWithCpns[dateIndex]['epeThresholds']
    
    # Plot diffs
    fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(15,6))
    # EPE
    ax1.set_title('Exposures by Threshold at date ' + str(atDate))
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('EPE')
    ax1.plot(thresholds, thresholdsEpeFormula, label="Formula")
    ax1.plot(thresholds, thresholdsEpeTest, label=getRegressorLabel(testRegressorName))
    ax1.legend()
    
    # Diffs
    ax2.set_title('Absolute Diffs vs Formula')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('EPE Diff')
    ax2.plot(thresholds, np.abs(thresholdsEpeTest-thresholdsEpeFormula), color="green", label=getRegressorLabel(testRegressorName))
    ax2.legend()
    
    plt.show()

    if addDiffsWithCoupons:

        fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(15,6))
        # EPE
        ax1.set_title('Exposures by Threshold at date ' + str(atDate) + ' - Coupons mode')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('EPE')
        ax1.plot(thresholds, thresholdsEpeFormula, label="Formula")
        ax1.plot(thresholds, thresholdsEpeTestWithCpns, label=getRegressorLabel(testRegressorName))
        ax1.legend()

        # Diffs
        ax2.set_title('Absolute Diffs vs Formula - Coupons mode')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('EPE Diff')
        ax2.plot(thresholds, np.abs(thresholdsEpeTestWithCpns-thresholdsEpeFormula), color="green", label=getRegressorLabel(testRegressorName))
        ax2.legend()

        plt.show()
