# -*- coding: utf-8 -*-

import numpy as np, scipy as sp
from math import sqrt, exp, ceil
# import sobol_seq

# simulations
def getSimulations2Steps(S, T1, T2, model, sigma, nbSamples):
    
    if model == 'ln':
        S1 = S * np.exp( - 0.5 * sigma**2 * T1 + sigma * sqrt(T1) * np.random.normal(size=nbSamples) ).reshape((nbSamples, 1))
        S2 = S1 * np.exp( - 0.5 * sigma**2 * (T2-T1) + sigma * sqrt(T2-T1) * np.random.normal(size=nbSamples) ).reshape((nbSamples, 1))
    else:
        S1 = S  + ( sigma * sqrt(T1) * np.random.normal(size=nbSamples) ).reshape((nbSamples, 1))
        S2 = S1 + ( sigma * sqrt(T2-T1) * np.random.normal(size=nbSamples) ).reshape((nbSamples, 1))
        
    return S1,S2

def getSimulations(S, T, step, model, sigma, nbSamples, useAntithetics):

    # generate simulation dates and steps
    dates = np.arange(0.0,T+step,step)
    dates[-1] = T
    dateFractions = dates[1:]-dates[:-1]

    # generate simulations
    simulations, factors = getSimulationsFromSteps(S,dateFractions,model,sigma,nbSamples,useAntithetics)
    return dates, simulations, factors
    
def getSimulationsFromSteps(S, steps, model, sigma, nbSamples, useAntithetics):

    nbSteps = len(steps)
    simulations = S * np.ones((nbSamples, nbSteps+1))
    factors = np.zeros((nbSamples, nbSteps+1))
    if abs(sigma) == .0:
        return simulations
        
    # get brownians
    if useAntithetics:
        # speedup using antithetics
        nbSimuls = int(.5*nbSamples)
        gaussians = np.random.normal(size= nbSimuls * nbSteps).reshape((nbSimuls,nbSteps))
        gaussians = np.concatenate((gaussians,-gaussians),axis=0)
    else:
        gaussians = np.random.normal(size= nbSamples * nbSteps).reshape((nbSamples,nbSteps))        
#        gaussians = sp.stats.norm.ppf(sobol_seq.i4_sobol_generate(nbSteps,nbSamples))

    brownians = np.cumsum( gaussians * np.sqrt(steps), axis=1)

    factors[:,1:] += brownians
    if model == 'ln':
        simulations[:,1:] *= np.exp( sigma * brownians - np.cumsum( 0.5 * sigma**2 * steps ) )
    else:
        simulations[:,1:] += sigma * brownians

    return simulations, factors

# evolve on a tim step by taking all the realisations applied to each 
# hence the resulting simulations are not independent across values at start of the period (but they are conditionnally on the starting point)
# this is to be used for fast chained regression
def generateDenseSetOnStep1(S, step, model, sigma, nbInnerSamples, useAntithetics):

    # generate brownians inner samples first
    if useAntithetics and nbInnerSamples!=1:
        # speedup using antithetics
        nbSimuls = int(.5*nbInnerSamples)
        gaussians = np.random.normal(size=nbSimuls).reshape((nbSimuls,1))
        gaussians = np.concatenate((gaussians,-gaussians),axis=0)
    else:
        gaussians = np.random.normal(size=nbInnerSamples).reshape((nbInnerSamples,1))
    
    # repeat them for all inputs
    denseS = np.repeat(S,nbInnerSamples,axis=0)
    denseGaussians = np.tile(gaussians,(S.shape[0],1)).reshape((-1,1))
    
    # build the densified simulation set
    if model == 'ln':
        return denseS * np.exp(sigma * denseGaussians * sqrt(step) - 0.5 * sigma**2 * step)
    else:
        return denseS + sigma * denseGaussians * sqrt(step)


def generateDenseSetOnStep(S, step, model, sigma, nbInnerSamples, useAntithetics):

    totalSamples = S.shape[0] * nbInnerSamples
    
    # generate brownians inner samples first
#    if useAntithetics and totalSamples!=1:
#        # speedup using antithetics
#        nbSimuls = int(.5*totalSamples)
#        gaussians = np.random.normal(size=nbSimuls).reshape((nbSimuls,1))
#        gaussians = np.concatenate((gaussians,-gaussians),axis=0)
#    else:
    gaussians = np.random.normal(size=totalSamples).reshape((totalSamples,1))
    
    # repeat them for all inputs
    denseS = np.repeat(S,nbInnerSamples,axis=0)
    denseGaussians = gaussians
    
    # build the densified simulation set
    if model == 'ln':
        return denseS * np.exp(sigma * denseGaussians * sqrt(step) - 0.5 * sigma**2 * step)
    else:
        return denseS + sigma * denseGaussians * sqrt(step)