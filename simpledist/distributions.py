__author__ = 'Timothy D. Morton <tim.morton@gmail.com>'
"""
Defines objects useful for describing probability distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import UnivariateSpline as interpolate
from scipy.integrate import quad
import numpy.random as rand
from scipy.special import erf
from scipy.optimize import leastsq

import pandas as pd

from plotutils import setfig

from .kde import KDE

#figure this generic loading thing out; draft stage currently
def load_distribution(filename,path=''):
    fns = pd.read_hdf(filename,path)
    store = pd.HDFStore(filename)
    if '{}/samples'.format(path) in store:
        samples = pd.read_hdf(filename,path+'/samples')
        samples = np.array(samples)
    minval = fns['vals'].iloc[0]
    maxval = fns['vals'].iloc[-1]
    pdf = interpolate(fns['vals'],fns['pdf'],s=0)
    cdf = interpolate(fns['vals'],fns['cdf'],s=0)

    attrs = store.get_storer('{}/fns'.format(path)).attrs
    keywords = attrs.keywords
    t = attrs.disttype
    store.close()
    return t.__init__()

    

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

    keywords : dict, optional
        Optional dictionary of keywords; these will be saved with the distribution
        when `save_hdf` is called.

    Raises
    ------
    ValueError
        If `cdf` is not provided and minval or maxval are infinity.
    
    """
    def __init__(self,pdf,cdf=None,name='',minval=-np.inf,maxval=np.inf,norm=None,
                 cdf_pts=500,keywords=None):
        self.name = name            
        self.pdf = pdf
        self.cdf = cdf 
        self.minval = minval
        self.maxval = maxval
        
        if keywords is None:
            self.keywords = {}
        else:
            self.keywords = keywords
        self.keywords['name'] = name
        self.keywords['minval'] = minval
        self.keywords['maxval'] = maxval


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
            cdf_fn = interpolate(pts,cdfgrid,s=0,k=1)
            
            def cdf(x):
                x = np.atleast_1d(x)
                y = np.atleast_1d(cdf_fn(x))
                y[np.where(x < self.minval)] = 0
                y[np.where(x > self.maxval)] = 1
                return y
            self.cdf = cdf
            #define minval_cdf, maxval_cdf 
            zero_mask = cdfgrid==0
            one_mask = cdfgrid==1
            if zero_mask.sum()>0:
                self.minval_cdf = pts[zero_mask][-1] #last 0 value
            if one_mask.sum()>0:
                self.maxval_cdf = pts[one_mask][0] #first 1 value
            
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
        if hasattr(self,'samples'):
            s = pd.Series(self.samples)
            s.to_hdf(filename,path+'/samples')
        store = pd.HDFStore(filename)
        attrs = store.get_storer('{}/fns'.format(path)).attrs
        attrs.keywords = self.keywords
        attrs.disttype = type(self)
        store.close()
    
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

        fig : None or int, optional
            Parameter to pass to `setfig`.  If `None`, then a new figure is 
            created; if a non-zero integer, the plot will go to that figure 
            (clearing everything first), if zero, then will overplot on 
            current axes.

        log : bool, optional
            If `True`, the x-spacing of the points to plot will be logarithmic.

        npoints : int, optional
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
        plt.ylim(ymin=0,ymax=self(xs).max()*1.2)

    def resample(self,N,minval=None,maxval=None,log=False,res=1e4):
        """Returns random samples generated according to the distribution

        Mirrors basic functionality of `rvs` method for `scipy.stats`
        random variates.  Implemented by mapping uniform numbers onto the
        inverse CDF using a closest-matching grid approach.

        Parameters
        ----------
        N : int
            Number of samples to return

        minval,maxval : float, optional
            Minimum/maximum values to resample.  Should both usually just be 
            `None`, which will default to `self.minval`/`self.maxval`.

        log : bool, optional
            Whether grid should be log- or linear-spaced.

        res : int, optional
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
        N = int(N)
        if minval is None:
            if hasattr(self,'minval_cdf'):
                minval = self.minval_cdf
            else:
                minval = self.minval
        if maxval is None:
            if hasattr(self,'maxval_cdf'):
                maxval = self.maxval_cdf
            else:
                maxval = self.maxval

        if maxval==np.inf or minval==-np.inf:
            raise ValueError('must have finite upper and lower bounds to resample. (set minval, maxval kws)')

        u = rand.random(size=N)
        if log:
            vals = np.logspace(log10(minval),log10(maxval),res)
        else:
            vals = np.linspace(minval,maxval,res)
            
        #sometimes cdf is flat.  so ys will need to be uniqued
        ys,yinds = np.unique(self.cdf(vals), return_index=True)
        vals = vals[yinds]
        

        inds = np.digitize(u,ys)
        return vals[inds]

    def rvs(self,*args,**kwargs):
        return self.resample(*args,**kwargs)

class Distribution_FromH5(Distribution):
    """Creates a Distribution object from one saved to an HDF file.

    File must have a `DataFrame` saved under [path]/fns in 
    the .h5 file, containing 'vals', 'pdf', and 'cdf' columns.
    If samples are saved in the HDF storer, then they will be restored
    to this object; so will any saved keyword attributes.

    These appropriate .h5 files will be created by a call to the `save_hdf` 
    method of the generic `Distribution` class.


    Parameters
    ----------
    filename : string
        .h5 file where the distribution is saved.

    path : string, optional
        Path within the .h5 file where the distribution is saved.  By 
        default this will be the root level, but can be anywhere.

    kwargs
        Keyword arguments are passed to the `Distribution` constructor.
    """
    def __init__(self,filename,path='',**kwargs):
        store = pd.HDFStore(filename,'r')
        fns = store[path+'/fns']
        if '{}/samples'.format(path) in store:
            samples = store[path+'/samples']
            self.samples = np.array(samples)
        minval = fns['vals'].iloc[0]
        maxval = fns['vals'].iloc[-1]
        pdf = interpolate(fns['vals'],fns['pdf'],s=0,k=1)
        
        #check to see if tabulated CDF is monotonically increasing
        d_cdf = fns['cdf'][1:] - fns['cdf'][:-1]
        if np.any(d_cdf < 0):
            logging.warning('tabulated CDF in {} is not strictly increasing. Recalculating CDF from PDF'.format(filename))
            cdf = None  #in this case, just recalc cdf from pdf
        else:
            cdf = interpolate(fns['vals'],fns['cdf'],s=0,k=1)
        Distribution.__init__(self,pdf,cdf,minval=minval,maxval=maxval,
                              **kwargs)

        store = pd.HDFStore(filename,'r')
        try:
            keywords = store.get_storer('{}/fns'.format(path)).attrs.keywords
            for kw,val in keywords.iteritems():
                setattr(self,kw,val)
        except AttributeError:
            logging.warning('saved distribution {} does not have keywords or disttype saved; perhaps this distribution was written with an older version.'.format(filename))
        store.close()


class Empirical_Distribution(Distribution):
    """Generates a Distribution object given a tabulated PDF.

    Parameters
    ----------
    xs : array-like
        x-values at which the PDF is evaluated

    pdf : array-like
        Values of pdf at provided x-values.

    smooth : int or float
        Smoothing parameter used by the interpolation.

    kwargs
        Keyword arguments passed to `Distribution` constructor.
    """
    def __init__(self,xs,pdf,smooth=0,**kwargs):
        pdf /= np.trapz(pdf,xs)
        fn = interpolate(xs,pdf,s=smooth)
        keywords = {'smooth':smooth}
        Distribution.__init__(self,fn,minval=xs.min(),maxval=xs.max(),
                              keywords=keywords,**kwargs)

class Gaussian_Distribution(Distribution):
    """Generates a normal distribution with given mu, sigma.

    ***It's probably better to use scipy.stats.norm rather than this
       if you care about numerical precision/speed and don't care about the
       plotting bells/whistles etc. the `Distribution` class provides.***

    Parameters
    ----------
    mu : float
        Mean of normal distribution.

    sig : float
        Width of normal distribution.

    kwargs
        Keyword arguments passed to `Distribution` constructor.
    """
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

        keywords = {'mu':self.mu,'sig':self.sig}

        Distribution.__init__(self,pdf,cdf,keywords=keywords,**kwargs)

    def __str__(self):
        return '%s = %.2f +/- %.2f' % (self.name,self.mu,self.sig)
        
    def resample(self,N,**kwargs):
        return rand.normal(size=int(N))*self.sig + self.mu


class Hist_Distribution(Distribution):
    """Generates a distribution from a histogram of provided samples.

    Uses `np.histogram` to create a histogram using the bins keyword,
    then interpolates this histogram to create the pdf to pass to the
    `Distribution` constructor.

    Parameters
    ----------
    samples : array-like
        The samples used to create the distribution

    bins : int or array-like, optional
        Keyword passed to `np.histogram`.  If integer, ths will be 
        the number of bins, if array-like, then this defines bin edges.

    equibin : bool, optional
        If true and ``bins`` is an integer ``N``, then the bins will be 
        found by splitting the data into ``N`` equal-sized groups.

    smooth : int or float
        Smoothing parameter used by the interpolation function.

    order : int
        Order of the spline to be used for interpolation.  Default is
        for linear interpolation.

    kwargs
        Keyword arguments passed to `Distribution` constructor.
    """
    def __init__(self,samples,bins=10,equibin=True,smooth=0,order=1,**kwargs):
        self.samples = samples
        
        if type(bins)==type(10) and equibin:
            N = len(samples)//bins
            sortsamples = np.sort(samples)
            bins = sortsamples[0::N]
            if bins[-1] != sortsamples[-1]:
                bins = np.concatenate([bins,np.array([sortsamples[-1]])])

        hist,bins = np.histogram(samples,bins=bins,density=True)
        self.bins = bins
        bins = (bins[1:] + bins[:-1])/2.
        pdf_initial = interpolate(bins,hist,s=smooth,k=order)
        def pdf(x):
            x = np.atleast_1d(x)
            y = pdf_initial(x)
            w = np.where((x < self.bins[0]) | (x > self.bins[-1]))
            y[w] = 0
            return y
        cdf = interpolate(bins,hist.cumsum()/hist.cumsum().max(),s=smooth,
                          k=order)

        if 'maxval' not in kwargs:
            kwargs['maxval'] = samples.max()
        if 'minval' not in kwargs:
            kwargs['minval'] = samples.min()

        keywords = {'bins':bins,'smooth':smooth,'order':order}

        Distribution.__init__(self,pdf,cdf,keywords=keywords,**kwargs)

    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.samples.mean(),self.samples.std())

    def plothist(self,fig=None,**kwargs):
        """Plots a histogram of samples using provided bins.
        
        Parameters
        ----------
        fig : None or int
            Parameter passed to `setfig`.

        kwargs
            Keyword arguments passed to `plt.hist`.
        """
        setfig(fig)
        plt.hist(self.samples,bins=self.bins,**kwargs)

    def resample(self,N):
        """Returns a bootstrap resampling of provided samples.

        Parameters
        ----------
        N : int
            Number of samples.
        """
        inds = rand.randint(len(self.samples),size=N)
        return self.samples[inds]

    def save_hdf(self,filename,path='',**kwargs):
        Distribution.save_hdf(self,filename,path=path,**kwargs)

    
class Box_Distribution(Distribution):
    """Simple distribution uniform between provided lower and upper limits.

    Parameters
    ----------
    lo,hi : float
        Lower/upper limits of the distribution.

    kwargs
        Keyword arguments passed to `Distribution` constructor.

    """
    def __init__(self,lo,hi,**kwargs):
        self.lo = lo
        self.hi = hi
        def pdf(x):
            return 1./(hi-lo) + 0*x
        def cdf(x):
            x = np.atleast_1d(x)
            y = (x - lo) / (hi - lo)
            y[x < lo] = 0
            y[x > hi] = 1
            return y

        Distribution.__init__(self,pdf,cdf,minval=lo,maxval=hi,**kwargs)

    def __str__(self):
        return '%.1f < %s < %.1f' % (self.lo,self.name,self.hi)

    def resample(self,N):
        """Returns a random sampling.
        """
        return rand.random(size=N)*(self.maxval - self.minval) + self.minval



############## Double LorGauss ###########

def double_lorgauss(x,p):
    """Evaluates a normalized distribution that is a mixture of a double-sided Gaussian and Double-sided Lorentzian.

    Parameters
    ----------
    x : float or array-like
        Value(s) at which to evaluate distribution

    p : array-like
        Input parameters: mu (mode of distribution),
                          sig1 (LH Gaussian width),
                          sig2 (RH Gaussian width),
                          gam1 (LH Lorentzian width),
                          gam2 (RH Lorentzian width),
                          G1 (LH Gaussian "strength"),
                          G2 (RH Gaussian "strength").

    Returns
    -------
    values : float or array-like
        Double LorGauss distribution evaluated at input(s).  If single value provided,
        single value returned. 
    """
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
    """Uses lmfit to fit a "Double LorGauss" distribution to a provided histogram.

    Uses a grid of starting guesses to try to avoid local minima.

    Parameters
    ----------
    bins, h : array-like
        Bins and heights of a histogram, as returned by, e.g., `np.histogram`.

    Ntry : int, optional
        Spacing of grid for starting guesses.  Will try `Ntry**2` different
        initial values of the "Gaussian strength" parameters `G1` and `G2`.

    Returns
    -------
    parameters : tuple
        Parameters of best-fit "double LorGauss" distribution.

    Raises
    ------
    ImportError
        If the lmfit module is not available.
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
    """Defines a "double LorGauss" distribution according to the provided parameters.

    Parameters
    ----------
    mu,sig1,sig2,gam1,gam2,G1,G2 : float
        Parameters of `double_lorgauss` function.

    kwargs
        Keyword arguments passed to `Distribution` constructor.
    """
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

        keywords = {'mu':mu,'sig1':sig1,
                    'sig2':sig2,'gam1':gam1,'gam2':gam2,
                    'G1':G1,'G2':G2}

        Distribution.__init__(self,pdf,keywords=keywords,**kwargs)
        
        
######## DoubleGauss #########

def doublegauss(x,p):
    """Evaluates normalized two-sided Gaussian distribution

    Parameters
    ----------
    x : float or array-like
        Value(s) at which to evaluate distribution

    p : array-like
        Parameters of distribution: (mu: mode of distribution,
                                     sig1: LH width,
                                     sig2: RH width)

    Returns
    -------
    value : float or array-like
        Distribution evaluated at input value(s).  If single value provided,
        single value returned.
    """
    mu,sig1,sig2 = p
    x = np.atleast_1d(x)
    A = 1./(np.sqrt(2*np.pi)*(sig1+sig2)/2.)
    ylo = A*np.exp(-(x-mu)**2/(2*sig1**2))
    yhi = A*np.exp(-(x-mu)**2/(2*sig2**2))
    y = x*0
    wlo = np.where(x < mu)
    whi = np.where(x >= mu)
    y[wlo] = ylo[wlo]
    y[whi] = yhi[whi]
    if np.size(x)==1:
        return y[0]
    else:
        return y    

def doublegauss_cdf(x,p):
    """Cumulative distribution function for two-sided Gaussian

    Parameters
    ----------
    x : float
        Input values at which to calculate CDF.

    p : array-like
        Parameters of distribution: (mu: mode of distribution,
                                     sig1: LH width,
                                     sig2: RH width)
    """
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
    """Fits a two-sided Gaussian to a set of samples.

    Calculates 0.16, 0.5, and 0.84 quantiles and passes these to
    `fit_doublegauss` for fitting.

    Parameters
    ----------
    samples : array-like
        Samples to which to fit the Gaussian.

    kwargs
        Keyword arguments passed to `fit_doublegauss`.
    """
    sorted_samples = np.sort(samples)
    N = len(samples)
    med = sorted_samples[N/2]
    siglo = med - sorted_samples[int(0.16*N)]
    sighi = sorted_samples[int(0.84*N)] - med
    return fit_doublegauss(med,siglo,sighi,median=True,**kwargs)
    

def fit_doublegauss(med,siglo,sighi,interval=0.683,p0=None,median=False,return_distribution=True):
    """Fits a two-sided Gaussian distribution to match a given confidence interval.

    The center of the distribution may be either the median or the mode.

    Parameters
    ----------
    med : float
        The center of the distribution to which to fit.  Default this
        will be the mode unless the `median` keyword is set to True.

    siglo : float
        Value at lower quantile (`q1 = 0.5 - interval/2`) to fit.  Often this is
        the "lower error bar."

    sighi : float
        Value at upper quantile (`q2 = 0.5 + interval/2`) to fit.  Often this is
        the "upper error bar."

    interval : float, optional
        The confidence interval enclosed by the provided error bars.  Default
        is 0.683 (1-sigma).

    p0 : array-like, optional
        Initial guess `doublegauss` parameters for the fit (`mu, sig1, sig2`).

    median : bool, optional
        Whether to treat the `med` parameter as the median or mode
        (default will be mode).

    return_distribution: bool, optional
        If `True`, then function will return a `DoubleGauss_Distribution` object.
        Otherwise, will return just the parameters.
    """
    if median:
        q1 = 0.5 - (interval/2)
        q2 = 0.5 + (interval/2)
        targetvals = np.array([med-siglo,med,med+sighi])
        qvals = np.array([q1,0.5,q2])
        def objfn(pars):
            logging.debug('{}'.format(pars))
            logging.debug('{} {}'.format(doublegauss_cdf(targetvals,pars),qvals))
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
    """A Distribution oject representing a two-sided Gaussian distribution

    This can be used to represent a slightly asymmetric distribution,
    and consists of two half-Normal distributions patched together at the
    mode, and normalized appropriately.  The pdf and cdf are according to
    the `doubleguass` and `doubleguass_cdf` functions, respectively.

    Parameters
    ----------
    mu : float
        The mode of the distribution.

    siglo : float
        Width of lower half-Gaussian.

    sighi : float
        Width of upper half-Gaussian.

    kwargs
        Keyword arguments are passed to `Distribution` constructor.
    """
    def __init__(self,mu,siglo,sighi,**kwargs):
        self.mu = mu
        self.siglo = float(siglo)
        self.sighi = float(sighi)
        def pdf(x):
            return doublegauss(x,(mu,siglo,sighi))
        def cdf(x):
            return doublegauss_cdf(x,(mu,siglo,sighi))
        
        if 'minval' not in kwargs:
            kwargs['minval'] = mu - 5*siglo
        if 'maxval' not in kwargs:
            kwargs['maxval'] = mu + 5*sighi

        keywords = {'mu':mu,'siglo':siglo,'sighi':sighi}

        Distribution.__init__(self,pdf,cdf,keywords=keywords,**kwargs)

    def __str__(self):
        return '%s = %.2f +%.2f -%.2f' % (self.name,self.mu,self.sighi,self.siglo)

    def resample(self,N,**kwargs):
        """Random resampling of the doublegauss distribution
        """
        lovals = self.mu - np.absolute(rand.normal(size=N)*self.siglo)
        hivals = self.mu + np.absolute(rand.normal(size=N)*self.sighi)

        u = rand.random(size=N)
        hi = (u < float(self.sighi)/(self.sighi + self.siglo))
        lo = (u >= float(self.sighi)/(self.sighi + self.siglo))

        vals = np.zeros(N)
        vals[hi] = hivals[hi]
        vals[lo] = lovals[lo]
        return vals
    
def powerlawfn(alpha,minval,maxval):
    C = powerlawnorm(alpha,minval,maxval)
    def fn(inpx):
        x = np.atleast_1d(inpx)
        y = C*x**(alpha)
        y[(x < minval) | (x > maxval)] = 0
        return y
    return fn

def powerlawnorm(alpha,minval,maxval):
    if np.size(alpha)==1:
        if alpha == -1:
            C = 1/np.log(maxval/minval)
        else:
            C = (1+alpha)/(maxval**(1+alpha)-minval**(1+alpha))
    else:
        C = np.zeros(np.size(alpha))
        w = np.where(alpha==-1)
        if len(w[0]>0):
            C[w] = 1./np.log(maxval/minval)*np.ones(len(w[0]))
            nw = np.where(alpha != -1)
            C[nw] = (1+alpha[nw])/(maxval**(1+alpha[nw])-minval**(1+alpha[nw]))
        else:
            C = (1+alpha)/(maxval**(1+alpha)-minval**(1+alpha))
    return C


class PowerLaw_Distribution(Distribution):
    def __init__(self,alpha,minval,maxval,**kwargs):
        self.alpha = alpha
        pdf = powerlawfn(alpha,minval,maxval)

        Distribution.__init__(self,pdf,minval=minval,maxval=maxval)


######## KDE ###########

class KDE_Distribution(Distribution):
    def __init__(self,samples,adaptive=True,draw_direct=True,bandwidth=None,**kwargs):
        self.samples = samples
        self.bandwidth = bandwidth
        self.kde = KDE(samples,adaptive=adaptive,draw_direct=draw_direct,
                       bandwidth=bandwidth)

        if 'minval' not in kwargs:
            kwargs['minval'] = samples.min()
        if 'maxval' not in kwargs:
            kwargs['maxval'] = samples.max()      

        keywords = {'adaptive':adaptive,'draw_direct':draw_direct,
                    'bandwidth':bandwidth}

        Distribution.__init__(self,self.kde,keywords=keywords,**kwargs)

    def save_hdf(self,filename,path='',**kwargs):
        Distribution.save_hdf(self,filename,path=path,**kwargs)

    def __str__(self):
        return '%s = %.1f +/- %.1f' % (self.name,self.samples.mean(),self.samples.std())

    def resample(self,N,**kwargs):
        return self.kde.resample(N,**kwargs)

class KDE_Distribution_Fromtxt(KDE_Distribution):
    def __init__(self,filename,**kwargs):
        samples = np.loadtxt(filename)
        KDE_Distribution.__init__(self,samples,**kwargs)


