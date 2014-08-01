import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

try:
    from plotutils import setfig
except ImportError:
    def setfig(fig,**kwargs):
        """
        Sets figure to 'fig' and clears; if fig is 0, does nothing (e.g. for overplotting)
    
        if fig is None (or anything else), creates new figure
        
        I use this for basically every function I write to make a plot.  I give the function
        a "fig=None" kw argument, so that it will by default create a new figure.
        """
        if fig:
            plt.figure(fig,**kwargs)
            plt.clf()
        elif fig==0:
            pass
        else:
            plt.figure(**kwargs)

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.integrate import quad
from scipy.stats import gaussian_kde
#from lmfit import minimize, Parameters, Parameter, report_fit
import numpy.random as rand
from scipy.special import erf
from scipy.optimize import leastsq

class Distribution(object):
    """Base class to describe probability distribution.
    
    Tries to have some functional overlap with scipy.stats random variates (e.g. ppf, rvs)
    """
    def __init__(self,pdf,cdf=None,name='',minval=-np.inf,maxval=np.inf,norm=None,
                 no_cdf=False,prior=None,cdf_pts=500):
        self.name = name

        def newpdf(x):
            if prior is None:
                return pdf(x)
            else:
                return prior(x)*pdf(x)

        self.lhood = pdf
            
        self.pdf = newpdf
        self.cdf = cdf #won't be right if prior given
        self.minval = minval
        self.maxval = maxval

        self.prior = prior

        if not hasattr(self,'Ndists'):
            self.Ndists = 1

        if norm is None:
            self.norm = quad(self.pdf,minval,maxval,full_output=1)[0]
        else:
            self.norm = norm

        if (cdf is None and not no_cdf and minval != -np.inf and maxval != np.inf) or prior is not None:
            pts = np.linspace(minval,maxval,cdf_pts)
            pdfgrid = self(pts)
            #fix special case: last value is infinity (i.e. from isotropic prior)
            if np.isinf(pdfgrid[-1]):
                tip_integral = quad(self,pts[-2],pts[-1])[0]
                #print tip_integral
                cdfgrid = pdfgrid[:-1].cumsum()/pdfgrid[:-1].cumsum().max() * (1-tip_integral)
                cdfgrid = np.concatenate((cdfgrid,[1]))
            else:
                cdfgrid = pdfgrid.cumsum()/pdfgrid.cumsum().max()

            cdf_fn = interpolate(pts,cdfgrid,s=0)
            def cdf(x):
                x = np.atleast_1d(x)
                y = np.atleast_1d(cdf_fn(x))
                y[np.where(x < self.minval)] = 0
                y[np.where(x > self.maxval)] = 1
                return y
            self.cdf = cdf

    def pctile(self,pct,res=1000):
        grid = np.linspace(self.minval,self.maxval,res)
        return grid[np.argmin(np.absolute(pct-self.cdf(grid)))]

    ppf = pctile

    def save_hdf(self,filename,path='',res=1000,logspace=False):
        if logspace:
            vals = np.logspace(np.log10(self.minval),
                               np.log10(self.maxval),
                               res)
        else:
            vals = np.linspace(self.minval,self.maxval,res)
        d = {'vals':vals,
             'pdf':self(vals),
             'cdf':self.cdf(vals)}
        df = pd.DataFrame(d)
        df.to_hdf(filename,path+'/fns')
    
    def __add__(self,other):
        return Combined_Distribution((self,other))

    def __radd__(self,other):
        return self.__add__(other)

    def __call__(self,x):
        y = self.pdf(x)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        w = np.where((x < self.minval) | (x > self.maxval))
        y[w] = 0
        return y/self.norm

    def __str__(self):
        return '%s = %.2f +%.2f -%.2f' % (self.name,
                                          self.pctile(0.5),
                                          self.pctile(0.84)-self.pctile(0.5),
                                          self.pctile(0.5)-self.pctile(0.16))

    def __repr__(self):
        return '<%s object: %s>' % (type(self),str(self))


    def plot(self,minval=None,maxval=None,fig=None,log=False,npts=500,plotprior=True,**kwargs):
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to plot. (set minval, maxval kws)')

        if log:
            xs = np.logspace(np.log10(minval),np.log10(maxval),npts)
        else:
            xs = np.linspace(minval,maxval,npts)

        setfig(fig)
        plt.plot(xs,self(xs),**kwargs)
        if plotprior:
            lhoodnorm = quad(self.lhood,self.minval,self.maxval)[0]
            plt.plot(xs,self.lhood(xs)/lhoodnorm,ls=':')
        plt.xlabel(self.name)
        plt.ylim(ymin=0)

    def resample(self,N,minval=None,maxval=None,log=False,res=1e4):
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to resample. (set minval, maxval kws)')

        u = rand.random(size=N)
        if log:
            vals = np.logspace(log10(minval),log10(maxval),res)
        else:
            vals = np.linspace(minval,maxval,res)
            
        ys = self.cdf(vals)
        inds = np.digitize(u,ys)
        return vals[inds]

    def rvs(self,*args,**kwargs):
        return self.resample(*args,**kwargs)

class Distribution_FromH5(Distribution):
    def __init__(self,filename,path='',disttype=None,**kwargs):
        """if disttype is 'hist' or 'kde' then samples are required
        """
        fns = pd.read_hdf(filename,path+'/fns')
        self.disttype = disttype
        if disttype in ['hist','kde']:
            samples = pd.read_hdf(filename,path+'/samples')
            self.samples = np.array(samples)
        minval = fns['vals'].iloc[0]
        maxval = fns['vals'].iloc[-1]
        pdf = interpolate(fns['vals'],fns['pdf'],s=0)
        cdf = interpolate(fns['vals'],fns['cdf'],s=0)
        Distribution.__init__(self,pdf,cdf,minval=minval,maxval=maxval,
                              **kwargs)


class EmpiricalDistribution(Distribution):
    def __init__(self,xs,pdf,smooth=0,**kwargs):
        pdf /= np.trapz(pdf,xs)
        fn = interpolate(xs,pdf,s=smooth)
        Distribution.__init__(self,fn,minval=xs.min(),maxval=xs.max())
        

class Gaussian_Distribution(Distribution):
    def __init__(self,mu,sig,**kwargs):
        self.mu = mu
        self.sig = sig
        def pdf(x):
            return 1./np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))

        def cdf(x):
            return 0.5*(1 + erf((x-mu)/np.sqrt(2*sig**2)))
            
        if 'minval' not in kwargs:
            kwargs['minval'] = mu - 10*sig
        if 'maxval' not in kwargs:
            kwargs['maxval'] = mu + 10*sig

        Distribution.__init__(self,pdf,cdf,**kwargs)

    def __str__(self):
        return '%s = %.2f +/- %.2f' % (self.name,self.mu,self.sig)
        
    def resample(self,N,**kwargs):
        return rand.normal(size=N)*self.sig + self.mu


class Hist_Distribution(Distribution):
    def __init__(self,samples,bins=10,smooth=0,order=1,**kwargs):
        self.samples = samples
        hist,bins = np.histogram(samples,bins=bins,density=True)
        self.bins = bins
        bins = (bins[1:] + bins[:-1])/2.
        pdf_initial = interpolate(bins,hist,s=smooth,k=order)
        def pdf(x):
            x = np.atleast_1d(x)
            y = pdf_initial(x)
            w = np.where((x < bins[0]) | (x > bins[-1]))
            y[w] = 0
            return y
        cdf = interpolate(bins,hist.cumsum()/hist.cumsum().max(),s=smooth,
                          k=order)

        if 'maxval' not in kwargs:
            kwargs['maxval'] = samples.max()
        if 'minval' not in kwargs:
            kwargs['minval'] = samples.min()

        Distribution.__init__(self,pdf,cdf,**kwargs)

    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.samples.mean(),self.samples.std())

    def plothist(self,fig=None,**kwargs):
        setfig(fig)
        plt.hist(self.samples,bins=self.bins,**kwargs)

    def resample(self,N):
        inds = rand.randint(len(self.samples),size=N)
        return self.samples[inds]

class Box_Distribution(Distribution):
    def __init__(self,lo,hi,**kwargs):
        self.lo = lo
        self.hi = hi
        def pdf(x):
            return 1./(hi-lo) + 0*x
        def cdf(x):
            x = np.atleast_1d(x)
            y = (x - lo) / (hi - lo)
            y[np.where(x < lo)] = 0
            y[np.where(x > hi)] = 1
            return y

        Distribution.__init__(self,pdf,cdf,minval=lo,maxval=hi,**kwargs)

    def __str__(self):
        return '%.1f < %s < %.1f' % (self.lo,self.name,self.hi)

    def resample(self,N):
        return rand.random(size=N)*(self.maxval - self.minval) + self.minval

class Combined_Distribution(Distribution):
    def __init__(self,dist_list,minval=-np.inf,maxval=np.inf,labels=None,**kwargs):
        self.dist_list = list(dist_list)
        #self.Ndists = len(dist_list)

        self.dict = {}
        if labels is not None:
            for label,dist in zip(labels,dist_list):
                self.dict[label] = dist

        N = 0
        
        for dist in dist_list:
            N += dist.Ndists
            
        self.Ndists = N
        self.minval = minval
        self.maxval = maxval

        def pdf(x):
            y = x*0
            for dist in dist_list:
                y += dist(x)
            return y/N

        Distribution.__init__(self,pdf,minval=minval,maxval=maxval,**kwargs)

    def __getitem__(self,ind):
        if type(ind) == type(1):
            return self.dist_list[ind]
        else:
            return self.dict[ind]



