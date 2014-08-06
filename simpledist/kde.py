import numpy as np
from scipy.stats import gaussian_kde
import numpy.random as rand
from scipy.integrate import quad

class KDE(object):
    """An implementation of a kernel density estimator allowing for adaptive kernels.

    If the `adaptive` keyword is set to `False`, then this will essentially be just
    a wrapper for the `scipy.stats.gaussian_kde` class.  If adaptive, though, it
    allows for different kernels and different kernel widths according to the
    "K-nearest-neighbors" algorithm as discussed `here <http://en.wikipedia.org/wiki/Variable_kernel_density_estimation#Balloon_estimators>`_.  The `fast` option does the NN calculation using
    broadcasting arrays rather than a brute-force sort.  By default the
    fast option will be used for datasets smaller than 5000.

    Parameters
    ----------
    dataset : array-like
        Data set from which to calculate the KDE.

    kernel : {'tricube','ep','gauss'}, optional
        Kernel function to use for adaptive estimator.

    adaptive : bool, optional
        Flag whether or not to use adaptive KDE.  If this is false, then this
        class will just be a wrapper for `scipy.stats.gaussian_kde`.

    k : `None` or int, optional
        Number to use for K-nearest-neighbor algorithm.  If `None`, then
        it will be set to the `N/4`, where `N` is the size of the dataset.

    fast : `None` or bool, optional
        If `None`, then `fast = N < 5001`, where `N` is the size of the dataset.
        `fast=True` will force array calculations, which will use lots of RAM
        if the dataset is large.

    norm : float, optional
        Allows the normalization of the distribution to be something other
        than unity

    bandwidth : `None` or float, optional
        Passed to `scipy.stats.gaussian_kde` if not using adaptive mode.

    weights : array-like, optional
        Not yet implemented.

    draw_direct : bool, optional
        If `True`, then resampling will be just a bootstrap resampling
        of the input samples.  If `False`, then resampling will actually
        resample each individual kernel (not recommended for large-ish
        datasets).
        
    kwargs
        Keyword arguments passed to `scipy.stats.gaussian_kde` if adaptive
        mode is not being used.
    """
    def __init__(self,dataset,kernel='tricube',adaptive=True,k=None,
                 fast=None,norm=1.,bandwidth=None,weights=None,
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

        self.norm=norm


        self.adaptive = adaptive
        self.fast = fast
        if adaptive:
            if self.fast==None:
                self.fast = self.n < 5001

            if self.fast:
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
            self.gauss_kde = gaussian_kde(self.dataset,bw_method=bandwidth,**kwargs)
            

    def renorm(self,norm):
        """Change the normalization"""
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
            
    def integrate_box(self,low,high,forcequad=False,**kwargs):
        """Integrates over a box. Optionally force quad integration, even for non-adaptive.

        If adaptive mode is not being used, this will just call the
        `scipy.stats.gaussian_kde` method `integrate_box_1d`.  Else,
        by default, it will call `scipy.integrate.quad`.  If the
        `forcequad` flag is turned on, then that integration will be
        used even if adaptive mode is off.

        Parameters
        ----------
        low : float
            Lower limit of integration

        high : float
            Upper limit of integration

        forcequad : bool
            If `True`, then use the quad integration even if adaptive mode is off.

        kwargs
            Keyword arguments passed to `scipy.integrate.quad`.
        """
        if not self.adaptive and not forcequad:
            return self.gauss_kde.integrate_box_1d(low,high)*self.norm
        return quad(self.evaluate,low,high,**kwargs)[0]

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

    draw = resample

def epkernel(u):
    x = np.atleast_1d(u)
    y = 3./4*(1-x*x)
    y[((x>1) | (x < -1))] = 0
    return y

def gausskernel(u):
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5*u*u)

def tricubekernel(u):
    x = np.atleast_1d(u)
    y = 35./32*(1-x*x)**3
    y[((x > 1) | (x < -1))] = 0
    return y

def kernelfn(kernel='tricube'):
    if kernel=='ep':
        return epkernel

    elif kernel=='gauss':
        return gausskernel

    elif kernel=='tricube':
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
