import numpy as np
from scipy.special import erf
from scipy.optimize import leastsq
import numpy.random as rand

from .distributions import Distribution

def doublegauss(x,p):
    mu,siglo,sighi = p
    x = np.atleast_1d(x)
    A = 1./(np.sqrt(2*np.pi)*(siglo+sighi)/2.)
    ylo = A*np.exp(-(x-mu)**2/(2*siglo**2))
    yhi = A*np.exp(-(x-mu)**2/(2*sighi**2))
    y = x*0
    wlo = np.where(x < mu)
    whi = np.where(x >= mu)
    y[wlo] = ylo[wlo]
    y[whi] = yhi[whi]
    return y    

def doublegauss_cdf(x,p):
    x = np.atleast_1d(x)
    mu,sig1,sig2 = p
    sig1 = np.absolute(sig1)
    sig2 = np.absolute(sig2)
    ylo = float(sig1)/(sig1 + sig2)*(1 + erf((x-mu)/np.sqrt(2*sig1**2)))
    yhi = float(sig1)/(sig1 + sig2) + float(sig2)/(sig1+sig2)*(erf((x-mu)/np.sqrt(2*sig2**2)))
    lo = x < mu
    hi = x >= mu
    return ylo*lo + yhi*hi

def fit_doublegauss_samples(samples,**kwargs):
    sorted_samples = np.sort(samples)
    N = len(samples)
    med = sorted_samples[N/2]
    siglo = med - sorted_samples[int(0.16*N)]
    sighi = sorted_samples[int(0.84*N)] - med
    return fit_doublegauss(med,siglo,sighi,median=True,**kwargs)
    

def fit_doublegauss(med,siglo,sighi,interval=0.68,p0=None,median=False,return_distribution=True,debug=False):
    if median:
        q1 = 0.5 - (interval/2)
        q2 = 0.5 + (interval/2)
        targetvals = np.array([med-siglo,med,med+sighi])
        qvals = np.array([q1,0.5,q2])
        def objfn(pars):
            if debug:
                print pars
                print doublegauss_cdf(targetvals,pars),qvals
            return doublegauss_cdf(targetvals,pars) - qvals

        if p0 is None:
            p0 = [med,siglo,sighi]
        pfit,success = leastsq(objfn,p0)

    else:
        q1 = 0.5 - (interval/2)
        q2 = 0.5 + (interval/2)
        targetvals = np.array([med-siglo,med+sighi])
        qvals = np.array([q1,q2])
        def objfn(pars):
            params = (med,pars[0],pars[1])
            return doublegauss_cdf(targetvals,params) - qvals

        if p0 is None:
            p0 = [siglo,sighi]
        pfit,success = leastsq(objfn,p0)
        pfit = (med,pfit[0],pfit[1])

    if return_distribution:
        dist = DoubleGauss_Distribution(*pfit)
        return dist
    else:
        return pfit
    
class DoubleGauss_Distribution(Distribution):
    def __init__(self,mu,siglo,sighi,**kwargs):
        self.mu = mu
        self.siglo = float(siglo)
        self.sighi = float(sighi)
        def pdf(x):
            #x = np.atleast_1d(x)
            #A = 1./(np.sqrt(2*np.pi)*(siglo+sighi)/2.)
            #ylo = A*np.exp(-(x-mu)**2/(2*siglo**2))
            #yhi = A*np.exp(-(x-mu)**2/(2*sighi**2))
            #y = x*0
            #wlo = np.where(x < mu)
            #whi = np.where(x >= mu)
            #y[wlo] = ylo[wlo]
            #y[whi] = yhi[whi]
            return doublegauss(x,(mu,siglo,sighi))
        def cdf(x):
            return doublegauss_cdf(x,(mu,siglo,sighi))
        
        if 'minval' not in kwargs:
            kwargs['minval'] = mu - 5*siglo
        if 'maxval' not in kwargs:
            kwargs['maxval'] = mu + 5*sighi

        Distribution.__init__(self,pdf,cdf,**kwargs)
        #Distribution.__init__(self,pdf,**kwargs)

    def __str__(self):
        return '%s = %.2f +%.2f -%.2f' % (self.name,self.mu,self.sighi,self.siglo)

    def resample(self,N,**kwargs):
        lovals = self.mu - np.absolute(rand.normal(size=N)*self.siglo)
        hivals = self.mu + np.absolute(rand.normal(size=N)*self.sighi)

        u = rand.random(size=N)
        whi = np.where(u < float(self.sighi)/(self.sighi + self.siglo))
        wlo = np.where(u >= float(self.sighi)/(self.sighi + self.siglo))

        vals = np.zeros(N)
        vals[whi] = hivals[whi]
        vals[wlo] = lovals[wlo]
        return vals
        
        return rand.normal(size=N)*self.sig + self.mu
