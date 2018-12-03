# -*- coding: utf-8 -*-

from functools import partial
import numpy as np, scipy as sp
from math import sqrt, exp

# payoff functions
def evalPayoff(payoffFn,T,S,model=None,sigma=None):
    
    # context for evaluating payoff at required date
    context = {
        'S': S,
        'T': T,
        'model':model,
        'sigma': sigma
    }
    
    # bound tool functions for this maturity
    toolFns = {
        'Fwd': S,
        'Call': partial(CallPrice,**context),
        'Put': partial(PutPrice,**context)
#        'Range': partial(RangePrice,**context)
    }
    
    return payoffFn(**toolFns)
    

def LogNormalCallPrice(K, S, T, sigma):
    d1 = (np.log(S/K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * sp.stats.norm.cdf(d1) - K * sp.stats.norm.cdf(d2)

def NormalCallPrice(K, S, T, sigma):
    moneyness = S-K
    sigmaSqrtT = sigma * sqrt(T)
    return sigmaSqrtT * sp.stats.norm.pdf(moneyness/sigmaSqrtT) + moneyness * sp.stats.norm.cdf(moneyness/sigmaSqrtT)

def CallPrice( K, S, T, model = 'ln', sigma = 0.0):
    
    if abs(T) == .0:
        return np.maximum(S-K,0.0)
    
    if model == 'ln':
        return LogNormalCallPrice(K,S,T,sigma)
    
    if model == 'n':
        return NormalCallPrice(K,S,T,sigma)

def PutPrice( K, S, T, model = 'ln', sigma = 0.0):
    
    return CallPrice( K, S, T, model, sigma) - S + K


#def LogNormalRangePrice(lowerBound, upperBound, S, T, sigma):
#
#    lowerProba = 0.0
#    if lowerBound != inf:
#        dL = (np.log(lowerBound/S) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
#        lowerProba = sp.stats.norm.cdf(dL)
#        
#    upperProba = 1.0
#    if upperBound != inf:
#        dU = (np.log(upperBound/S) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
#        upperProba = sp.stats.norm.cdf(dU)
#
#    return upperProba - lowerProba
#
#def RangePrice( lowerBound, upperBound, S, T, model = 'ln', sigma = 0.0):
#    
#    if abs(T) == .0:
#        return (S>=lowerBound) * (S<=upperBound)
#    
#    if model == 'ln':
#        return LogNormalRangePrice(lowerBound,upperBound,S,T,sigma)
#    
#    if model == 'n':
#        raise Exception('not implemented yet')
##         return NormalRangePrice(lowerBound,upperBound,S,T,sigma)

# cva calc tool
def cva(dates,epe,intensity,recovery):
    dateFractions = dates[1:] - dates[:-1]
    return (recovery-1) * intensity * np.sum( np.exp(-intensity*dates[1:]) * dateFractions * epe)

def getAllExposures(exposures, exposuresName):
    return [x[exposuresName] for x in exposures]

