import numpy as np

from .distributions import Distribution


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
