import numpy as np
import mpmath as mpm

def rho(v, D):
    return v/(2*D)

def omega(v, D, s):
    return np.sqrt(v**2 + (4*D*s))

def theta(v, D, s):
    return omega(v, D, s)/(2*D)

def alpha_plus(v, D, s):
    return rho(v, D) + theta(v, D, s)

def alpha_minus(v, D, s):
    return rho(v, D) - theta(v, D, s)

def lt_fptd0(v, D, s, x0):
    #Equation 2.22 of https://doi.org/10.48550/arXiv.2311.03939
    num1 = mpm.exp(-rho(v,D)*x0)
    num2a = omega(v,D,s)*mpm.cosh(theta(v,D,s)*(x0-1))
    num2b = v*mpm.sinh(theta(v,D,s)*(x0-1))
    
    den1 = omega(v,D,s)*mpm.cosh(theta(v,D,s))
    den2 = v*mpm.sinh(theta(v,D,s))
    
    return num1*((num2a+num2b)/(den1-den2))

def lt_fptdr(v, D, s, x0, r):
    #Equation 3.1 of https://doi.org/10.48550/arXiv.2311.03939
    num = (s+r)*lt_fptd0(v,D,s+r,x0)
    den = s + r*lt_fptd0(v,D,s+r,x0)
    
    return num/den

def lt_fptd(v,D,s,x0,r):
    if r>0:
        return lt_fptdr(v,D,s,x0,r)
    else:
        return lt_fptd0(v,D,s,x0)
    
def moments(v, D, x0, r):
    """
    Computes for the analytical moments from the Laplace-transformed FPT distribution
    Taken from Magalang, et. al (https://doi.org/10.48550/arXiv.2311.03939)

    Parameters
    ----------
    v : float
        Drift constant.
    D : float
        Diffusion constant.
    x0 : float
        Initial position. Must be from 0 <= x0 < 1.
    r : float
        Resetting rate.

    Returns
    -------
    mfpt : float
        Analytical mean.
    vfpt : float

    """
    lt_fpt = lambda s: lt_fptd(v, D, s, x0, r)
    mfpt = float(-mpm.diff(lt_fpt, 0).real)
    vfpt = float(mpm.diff(lt_fpt, 0, n = 2).real)
    return mfpt, vfpt

def loss_fcn(params, het_mfpt):
    """
    Take the absolute difference of the heterogeneous model mean RDT and the analytical mean RDT following the SDE in Eq. 14

    Parameters
    ----------
    params : list
        v : float. Drift constant.
        D : float. Diffusion constant.
    het_mfpt : float
        Mean RDT of the heterogeneous function.

    Returns
    -------
    loss : float
        Absolute mean difference.
    eul_vfpt : float
        Analytical variance of the drift-diffusion process following the SDE in Eq. 14.

    """
    v, D = params
    
    try:
        eul_mfpt, eul_vfpt = moments(v, D, 0.8, 0)
        loss = np.abs(het_mfpt-eul_mfpt)
    except:
        eul_mfpt = np.nan 
        eul_vfpt = np.nan 
        loss = np.nan
    
    return loss, eul_vfpt