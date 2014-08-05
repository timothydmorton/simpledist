__author__ = 'Timothy D. Morton <tim.morton@gmail.com>'
"""
Defines objects useful for describing probability distributions.
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from plotutils import setfig

from scipy.interpolate import UnivariateSpline as interpolate
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import numpy.random as rand
from scipy.special import erf
from scipy.optimize import leastsq

class Distribution(object):
    """Base class to describe probability distribution.
    
    Has some minimal functional overlap with scipy.stats random variates
    (e.g. `ppf`, `rvs`)
        
    Parameters
    ----------
    pdf : callable
        The probability density function to be used.  Does not have to be
        normalized, but must be non-negative.
        
    cdf : callable, optional
        The cumulative distribution function.  If not provided, this will
        be tabulated from the pdf, as long as minval and maxval are also provided
        
    name : string, optional
        The name of the distribution (will be used, for example, to label a plot).
        Default is empty string.
        
    minval,maxval : float, optional
        The minimum and maximum values of the distribution.  The Distribution will
        evaluate to zero outside these ranges, and this will also define the range
        of the CDF.  Defaults are -np.inf and +np.inf.  If these are not explicity
        provided, then a CDF function must be provided.

    norm : float, optional
        If not provided, this will be calculated by integrating the pdf from
        minval to maxval so that the Distribution is a proper PDF that integrates
        to unity.  `norm` can be non-unity if desired, but beware, as this will
        cause some things to act unexpectedly.

    cdf_pts : int, optional
        Number of points to tabulate in order to calculate CDF, if not provided.
        Default is 500.

    Raises
    ------
    ValueError
        If `cdf` is not provided and minval or maxval are infinity.
    
    """
    def __init__(self,pdf,cdf=None,name='',minval=-np.inf,maxval=np.inf,norm=None,
                 cdf_pts=500):
        self.name = name            
        self.pdf = pdf
        self.cdf = cdf 
        self.minval = minval
        self.maxval = maxval

        if norm is None:
            self.norm = quad(self.pdf,minval,maxval,full_output=1)[0]
        else:
            self.norm = norm

        if cdf is None and (minval == -np.inf or maxval == np.inf):
            raise ValueError('must provide either explicit cdf function or explicit min/max values')

        else: #tabulate & interpolate CDF.
            pts = np.linspace(minval,maxval,cdf_pts)
            pdfgrid = self(pts)
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
        """Returns the desired percentile of the distribution.

        Will only work if properly normalized.  Designed to mimic
        the `ppf` method of the `scipy.stats` random variate objects.
        Works by gridding the CDF at a given resolution and matching the nearest
        point.  NB, this is of course not as precise as an analytic ppf.

        Parameters
        ----------

        pct : float
            Percentile between 0 and 1.

        res : int, optional
            The resolution at which to grid the CDF to find the percentile.

        Returns
        -------
        percentile : float
        """
        grid = np.linspace(self.minval,self.maxval,res)
        return grid[np.argmin(np.absolute(pct-self.cdf(grid)))]

    ppf = pctile

    def save_hdf(self,filename,path='',res=1000,logspace=False):
        """Saves distribution to an HDF5 file.

        Saves a pandas `dataframe` object containing tabulated pdf and cdf
        values at a specfied resolution.  After saving to a particular path, a
        distribution may be regenerated using the `Distribution_FromH5` subclass.  

        Parameters
        ----------
        filename : string
            File in which to save the distribution.  Should end in .h5.

        path : string, optional
            Path in which to save the distribution within the .h5 file.  By
            default this is an empty string, which will lead to saving the
            `fns` dataframe at the root level of the file.

        res : int, optional
            Resolution at which to grid the distribution for saving.

        logspace : bool, optional
            Sets whether the tabulated function should be gridded with log or
            linear spacing.  Default will be logspace=False, corresponding
            to linear gridding.
        """
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
    
    def __call__(self,x):
        """
        Evaluates pdf.  Forces zero outside of (self.minval,self.maxval).  Will return

        Parameters
        ----------
        x : float, array-like
            Value(s) at which to evaluate PDF.

        Returns
        -------
        pdf : float, array-like
            Probability density (or re-normalized density if self.norm was explicity
            provided.
        
        """
        y = self.pdf(x)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        y[(x < self.minval) | (x > self.maxval)] = 0
        y /= self.norm
        if np.size(x)==1:
            return y[0]
        else:
            return y
        
    def __str__(self):
        return '%s = %.2f +%.2f -%.2f' % (self.name,
                                          self.pctile(0.5),
                                          self.pctile(0.84)-self.pctile(0.5),
                                          self.pctile(0.5)-self.pctile(0.16))

    def __repr__(self):
        return '<%s object: %s>' % (type(self),str(self))


    def plot(self,minval=None,maxval=None,fig=None,log=False,
             npts=500,**kwargs):
        """
        Plots distribution.

        Parameters
        ----------
        minval : float,optional
            minimum value to plot.  Required if minval of Distribution is 
            `-np.inf`.

        maxval : float, optional
            maximum value to plot.  Required if maxval of Distribution is 
            `np.inf`.

        fig : None or int
            Parameter to pass to `setfig`.  If `None`, then a new figure is 
            created; if a non-zero integer, the plot will go to that figure 
            (clearing everything first), if zero, then will overplot on 
            current axes.

        log : bool
            If `True`, the x-spacing of the points to plot will be logarithmic.

        npoints : int
            Number of points to plot.

        kwargs
            Keyword arguments are passed to plt.plot

        Raises
        ------
        ValueError
            If finite lower and upper bounds are not provided.
        """
        if minval is None:
            minval = self.minval
        if maxval is None:
            maxval = self.maxval
        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to plot. (use minval, maxval kws)')

        if log:
            xs = np.logspace(np.log10(minval),np.log10(maxval),npts)
        else:
            xs = np.linspace(minval,maxval,npts)

        setfig(fig)
        plt.plot(xs,self(xs),**kwargs)
        plt.xlabel(self.name)
        plt.ylim(ymin=0)

    def resample(self,N,minval=None,maxval=None,log=False,res=1e4):
        """Returns random samples generated according to the distribution

        Mirrors basic functionality of `rvs` method for `scipy.stats`
        random variates.  Implemented by mapping uniform numbers onto the
        inverse CDF using a closest-matching grid approach.

        Parameters
        ----------
        N : int
            Number of samples to return

        minval,maxval : float
            Minimum/maximum values to resample.  Should both usually just be 
            `None`, which will default to `self.minval`/`self.maxval`.

        log : bool
            Whether grid should be log- or linear-spaced.

        res : int
            Resolution of CDF grid used.

        Returns
        -------
        values : ndarray
            N samples.

        Raises
        ------
        ValueError
            If maxval/minval are +/- infinity, this doesn't work because of
            the grid-based approach.

        """
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
    """Creates a Distribution object from one saved to an HDF file.

    File must have a `DataFrame` saved under [path]/fns in 
    the .h5 file, containing 'vals', 'pdf', and 'cdf' columns.  If the
    `disttype` keyword is set to 'hist' or 'kde', then also [path]/samples
    should exist in the .h5 file containing the array of samples.  These
    appropriate .h5 files will be created by a call to the `save_hdf` method
    of the generic `Distribution` class.

    Parameters
    ----------
    filename : string
        .h5 file where the distribution is saved.

    path : string
        Path within the .h5 file where the distribution is saved.  By 
        default this will be the root level, but can be anywhere.

    disttype : {None, 'hist', 'kde'}
        If this is set to 'hist' or 'kde', then samples must also be 
        saved in the .h5 file.  (This is done automatically when saving
        `Hist_Distribution` or `KDE_Distribution` objects.

    kwargs
        Keyword arguments are passed to the `Distribution` constructor.
    """
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


class Empirical_Distribution(Distribution):
    """Generates a Distribution object given a tabulated PDF.

    Parameters
    ----------
    
    """
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



############## Double LorGauss ###########

def double_lorgauss(x,p):
    mu,sig1,sig2,gam1,gam2,G1,G2 = p
    gam1 = float(gam1)
    gam2 = float(gam2)

    G1 = abs(G1)
    G2 = abs(G2)
    sig1 = abs(sig1)
    sig2 = abs(sig2)
    gam1 = abs(gam1)
    gab2 = abs(gam2)
    
    L2 = (gam1/(gam1 + gam2)) * ((gam2*np.pi*G1)/(sig1*np.sqrt(2*np.pi)) - 
                                 (gam2*np.pi*G2)/(sig2*np.sqrt(2*np.pi)) +
                                 (gam2/gam1)*(4-G1-G2))
    L1 = 4 - G1 - G2 - L2

    
    #print G1,G2,L1,L2
    
    y1 = G1/(sig1*np.sqrt(2*np.pi)) * np.exp(-0.5*(x-mu)**2/sig1**2) +\
      L1/(np.pi*gam1) * gam1**2/((x-mu)**2 + gam1**2)
    y2 = G2/(sig2*np.sqrt(2*np.pi)) * np.exp(-0.5*(x-mu)**2/sig2**2) +\
      L2/(np.pi*gam2) * gam2**2/((x-mu)**2 + gam2**2)
    lo = (x < mu)
    hi = (x >= mu)
        
    return  y1*lo + y2*hi

def fit_double_lorgauss(bins,h,Ntry=5):
    """Uses lmfit to fit a "Double LorGauss" distribution.
    """
    try:
        from lmfit import minimize, Parameters, Parameter, report_fit
    except ImportError:
        raise ImportError('you need lmfit to use this function.')
        
    #make sure histogram is normalized
    h /= np.trapz(h,bins)

    #zero-pad the ends of the distribution to keep fits positive
    N = len(bins)
    dbin = (bins[1:]-bins[:-1]).mean()
    newbins = np.concatenate((np.linspace(bins.min() - N/10*dbin,bins.min(),N/10),
                             bins,
                             np.linspace(bins.max(),bins.max() + N/10*dbin,N/10)))
    newh = np.concatenate((np.zeros(N/10),h,np.zeros(N/10)))

    
    mu0 = bins[np.argmax(newh)]
    sig0 = abs(mu0 - newbins[np.argmin(np.absolute(newh - 0.5*newh.max()))])

    def set_params(G1,G2):
        params = Parameters()
        params.add('mu',value=mu0)
        params.add('sig1',value=sig0)
        params.add('sig2',value=sig0)
        params.add('gam1',value=sig0/10)
        params.add('gam2',value=sig0/10)
        params.add('G1',value=G1)
        params.add('G2',value=G2)
        return params

    sum_devsq_best = np.inf
    outkeep = None
    for G1 in np.linspace(0.1,1.9,Ntry):
        for G2 in np.linspace(0.1,1.9,Ntry):
            params = set_params(G1,G2)
        
            def residual(ps):
                pars = (params['mu'].value,
                        params['sig1'].value,
                        params['sig2'].value,
                        params['gam1'].value,
                        params['gam2'].value,
                        params['G1'].value,
                        params['G2'].value)
                hmodel = double_lorgauss(newbins,pars)
                return newh-hmodel

            out = minimize(residual,params)
            pars = (out.params['mu'].value,out.params['sig1'].value,
                    out.params['sig2'].value,out.params['gam1'].value,
                    out.params['gam2'].value,out.params['G1'].value,
                    out.params['G2'].value)

            sum_devsq = ((newh - double_lorgauss(newbins,pars))**2).sum()
            #print 'devs = %.1f; initial guesses for G1, G2; %.1f, %.1f' % (sum_devsq,G1, G2)
            if sum_devsq < sum_devsq_best:
                sum_devsq_best = sum_devsq
                outkeep = out
            

    return (outkeep.params['mu'].value,abs(outkeep.params['sig1'].value),
            abs(outkeep.params['sig2'].value),abs(outkeep.params['gam1'].value),
            abs(outkeep.params['gam2'].value),abs(outkeep.params['G1'].value),
            abs(outkeep.params['G2'].value))


class DoubleLorGauss_Distribution(Distribution):
    def __init__(self,mu,sig1,sig2,gam1,gam2,G1,G2,**kwargs):
        self.mu = mu
        self.sig1 = sig1
        self.sig2 = sig2
        self.gam1 = gam1
        self.gam2 = gam2
        self.G1 = G1
        #self.L1 = L1
        self.G2 = G2
        #self.L2 = L2
        def pdf(x):
            return double_lorgauss(x,(self.mu,self.sig1,self.sig2,
                                      self.gam1,self.gam2,
                                      self.G1,self.G2,))

        Distribution.__init__(self,pdf,**kwargs)
        
        
######## DoubleGauss #########

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


######## KDE ###########

class KDE_Distribution(Distribution):
    def __init__(self,samples,adaptive=True,draw_direct=True,bandwidth=None,**kwargs):
        self.samples = samples
        self.kde = KDE(samples,adaptive=adaptive,draw_direct=draw_direct,
                       bandwidth=bandwidth)

        if 'minval' not in kwargs:
            kwargs['minval'] = samples.min()
        if 'maxval' not in kwargs:
            kwargs['maxval'] = samples.max()      

        Distribution.__init__(self,self.kde,**kwargs)

    def save_hdf(self,filename,path=''):
        s = pd.Series(self.samples)
        s.to_hdf(filename,path+'/samples')
        Distribution.save_hdf(self,filename,path=path)
        
    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.samples.mean(),self.samples.std())

    def resample(self,N,**kwargs):
        return self.kde.resample(N,**kwargs)

class KDE_Distribution_Fromtxt(KDE_Distribution):
    def __init__(self,filename,**kwargs):
        samples = np.loadtxt(filename)
        KDE_Distribution.__init__(self,samples,**kwargs)


class KDE(object):
    def __init__(self,dataset,kernel='tricube',adaptive=True,k=None,lo=None,hi=None,\
                     fast=None,norm=None,bandwidth=None,weights=None,
                 draw_direct=False,**kwargs):
        self.dataset = np.atleast_1d(dataset)
        self.weights = weights
        self.n = np.size(dataset)
        self.kernel = kernelfn(kernel)
        self.kernelname = kernel
        self.bandwidth = bandwidth
        self.draw_direct = draw_direct
        if k:
            self.k = k
        else:
            self.k = self.n/4

        if not norm:
            self.norm=1.
        else:
            self.norm=norm


        self.adaptive = adaptive
        self.fast = fast
        if adaptive:
            if fast==None:
                fast = self.n < 5001

            if fast:
                #d1,d2 = np.meshgrid(self.dataset,self.dataset) #use broadcasting instead of meshgrid
                diff = np.absolute(self.dataset - self.dataset[:,np.newaxis])
                diffsort = np.sort(diff,axis=0)
                self.h = diffsort[self.k,:]

        ##Attempt to handle larger datasets more easily:
            else:
                sortinds = np.argsort(self.dataset)
                x = self.dataset[sortinds]
                h = np.zeros(len(x))
                for i in np.arange(len(x)):
                    lo = i - self.k
                    hi = i + self.k + 1
                    if lo < 0:
                        lo = 0
                    if hi > len(x):
                        hi = len(x)
                    diffs = abs(x[lo:hi]-x[i])
                    h[sortinds[i]] = np.sort(diffs)[self.k]
                self.h = h
        else:
            self.gauss_kde = gaussian_kde(self.dataset,bw_method=bandwidth)
            
        self.properties=dict()

        self.lo = lo
        self.hi = hi

    def shifted(self,x):
        new = kde(self.dataset+x,self.kernel,self.adaptive,self.k,self.lo,self.hi,self.fast,self.norm)
        return new

    def renorm(self,norm):
        self.norm = norm

    def evaluate(self,points):
        if not self.adaptive:
            return self.gauss_kde(points)*self.norm
        points = np.atleast_1d(points).astype(self.dataset.dtype)
        k = self.k

        npts = np.size(points)

        h = self.h
        
        X,Y = np.meshgrid(self.dataset,points)
        H = np.resize(h,(npts,self.n))

        U = (X-Y)/H.astype(float)

        result = 1./self.n*1./H*self.kernel(U)
        return np.sum(result,axis=1)*self.norm
            
    __call__ = evaluate
            
    def __imul__(self,factor):
        self.renorm(factor)
        return self

    #def __add__(self,other):
    #    return composite_kde(self,other)

    #__radd__ = __add__

    def integrate_box(self,low,high,npts=500,forcequad=False):
        if not self.adaptive and not forcequad:
            return self.gauss_kde.integrate_box_1d(low,high)*self.norm
        pts = np.linspace(low,high,npts)
        return quad(self.evaluate,low,high)[0]

    def draw(self,size=None):
        return self.resample(size)

    def resample(self,size=None,direct=None):
        if direct is None:
            direct = self.draw_direct
        size=int(size)

        if not self.adaptive:
            return np.squeeze(self.gauss_kde.resample(size=size))

        if direct:
            inds = rand.randint(self.n,size=size)
            return self.dataset[inds]
        else:
            if size is None:
                size = self.n
            indices = rand.randint(0,self.n,size=size)
            means = self.dataset[indices]
            h = self.h[indices]
            fuzz = kerneldraw(size,self.kernelname)*h
            return np.squeeze(means + fuzz)
    

def epkernel(u):
    x = np.atleast_1d(u)
    y = 3./4*(1-x*x)
    y[np.where((x>1) | (x < -1))] = 0
    return y

def gausskernel(u):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5*u*u)

def tricubekernel(u):
    x = np.atleast_1d(u)
    y = 35./32*(1-x*x)**3
    y[np.where((x > 1) | (x < -1))] = 0
    return y

def kernelfn(kernel='tricube'):
    if kernel=='ep':
        #def fn(u):
        #    x = atleast_1d(u)
        #    y = 3./4*(1-x*x)
        #    y[where((x>1) | (x<-1))] = 0
        #    return y
        #return fn
        return epkernel

    elif kernel=='gauss':
        #return lambda x: 1/sqrt(2*pi)*exp(-0.5*x*x)
        return gausskernel

    elif kernel=='tricube':
        #def fn(u):
        #    x = atleast_1d(u)
        #    y = 35/32.*(1-x*x)**3
        #    y[where((x>1) | (x<-1))] = 0
        #    return y
        #return fn
        return tricubekernel

def kerneldraw(size=1,kernel='tricube',exact=False):
    if kernel=='tricube':
        fn = lambda x: 1./2 + 35./32*x - 35./32*x**3 + 21./32*x**5 - 5./32*x**7
        u = rand.random(size=size)

        if not exact:
            xs = np.linspace(-1,1,1e4)
            ys = fn(xs)
        
            inds = np.digitize(u,ys)
            return xs[inds]
        else:
            #old way (exact)
            rets = np.zeros(size)
            for i in np.arange(size):
                f = lambda x: u[i]-fn(x)
                rets[i] = newton(f,0,restrict=(-1,1))
            return rets

def deriv(f,c,dx=0.0001):
    """
    deriv(f,c,dx)  --> float
    
    Returns f'(x), computed as a symmetric difference quotient.
    """
    return (f(c+dx)-f(c-dx))/(2*dx)

def fuzzyequals(a,b,tol=0.0001):
    return abs(a-b) < tol

def newton(f,c,tol=0.0001,restrict=None):
    """
    newton(f,c) --> float
    
    Returns the x closest to c such that f(x) = 0
    """
    #print c
    if restrict:
        lo,hi = restrict
        if c < lo or c > hi:
            print c
            c = random*(hi-lo)+lo

    if fuzzyequals(f(c),0,tol):
        return c
    else:
        try:
            return newton(f,c-f(c)/deriv(f,c,tol),tol,restrict)
        except:
            return None
