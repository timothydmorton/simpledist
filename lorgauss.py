import numpy as np

from .distributions import Distribution

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
        
        
