from scipy.special import jv as Jv
from scipy.special import yv as Yv
import scipy.integrate as scint
import numpy as np

def cont_ana_MRDT(params,r):
    """
    Analytical mean RDT of the coupled continuous model, given by Eq. 6 of the main text

    Parameters
    ----------
    params : list
        List of input parameters, in order:
            D: Diffusion constant
            v: Drift constant
            WR: Reset rate, 1/tau
            d: Number of therapies, N_T
            R1: Position of the reflecting boundary
            R0: Position of the absorbing boundary.
    r : float
        Initial therapy efficacy.

    Returns
    -------
    mrdt : float
        Mean resistance development time of the coupled continuous model.

    """
    D,v,WR,d,R1,R0=params
    if (WR==0):
        a=(v/D+d)
        if a==2:
            return ((R0**2-r**2)/4+R1**2*(np.log(r/R0))/2)/D
        else:
            dum=2-a
            return ((R0**2-r**2)/2+R1**(int(a))*(r**dum-R0**dum)/dum)/(a*D)
    else:
        k=np.sqrt(WR/D)
        ws=-complex(0,k*r)
        w0=-complex(0,k*R0);w1=-complex(0,k*R1)
        a=(v/D+d-1)
        dumr=(r/R0)**((a-1)/2)
        den=(Jv((a+1)/2,w1)*Yv((a-1)/2,ws)-Jv((a-1)/2,ws)*Yv((a+1)/2,w1))*WR
        num=Jv((a+1)/2,w1)*(-Yv((a-1)/2,ws) +dumr* Yv((a-1)/2,w0)) +  (Jv((a-1)/2,ws) -  dumr*Jv((a-1)/2,w0))*  Yv((a+1)/2,w1)
        return np.real(num/den)
    
def cont_sim_RDT(params,r0, res_type = "unlimited", AT = 0, trajectory = False):
    """
    Generate one trajectory for rotationally invariant diffusive particle using spherical coordinates using the Euler-Maruyama algorithm. Follows the SDE in Eq. 5 of the main text.

    Parameters
    ----------
    params : list
        List of input parameters, in order:
            D: Diffusion constant
            v: Drift constant
            WR: Reset rate, 1/tau
            d: Number of therapies, N_T
            R1: Position of the reflecting boundary
            R0: Position of the absorbing boundary.
    r0 : float
        Initial therapy efficacy.
    res_type : "unlimited", "limited", "costed", optional
        "unlimited" : No constraints in therapy switching
        "limited" : Explicit limit on therapy switching
        "costed" : Subsequent switches impose a cost, following Eq. 12 with c = 10**(d-1)
        The default is "unlimited".
    AT : integer, optional
        Used when res_type = "limited". Allowed limit of therapy switching, described in Eq. 11
        The default is 0.
    trajectory : boolean, optional
        If true, returns the trajectory of the process. The default is False.
    Returns
    -------
    if history = True:
        t : array
            Time array of the trajectory.
        r : array
            Position array of the trajectory (in radial distances).
    else:
        t : float
            Time of the trajectory at termination (RDT)
    """
    
    D,v,WR,d,R1,R0=params
    dt=0.01/np.sqrt(D)
    sqdh2=np.sqrt(2.0*D*dt)

    #Initial conditions
    r=r0
    t=0
    if trajectory:
        rs=[r0];ts=[t]

    #Updates reset time
    if WR > 0:
        tR=-1/WR*np.log(np.random.uniform()) #Time for first reset
    else:
        tR = np.inf
        
    clock=0 # Dummy variable to indicate if I reached reset time (tR)
    gamma=0 # number of resets along trajectory
    lim = int((AT/d) - 1)
    WR0 = WR
    while r > R0:
        t+=dt
        clock+=dt
        if(clock>tR):
            gamma += 1
            if res_type == "limited":
                lim -= 1
                if lim > 0:
                    tR=-1/WR*np.log(np.random.uniform()) # Time for next reset
                else:
                    tR = np.inf
            elif res_type == "costed":
                WR = WR0*np.exp(-(10**(d-1))*WR0*(gamma))
                tR=-1/WR*np.log(np.random.uniform())
            elif res_type == "unlimited":
                tR=-1/WR*np.log(np.random.uniform()) # Time for next reset
            
            clock=0
            r=r0
        else:
            dr=(v/D+d-1)*D*dt/r+sqdh2*np.random.normal()
            rn=r+dr
            if(r>R1):
                r=2*R1-rn
            else:
                r=rn
        if trajectory:
            rs.append(r);ts.append(t)
        
    if trajectory:
        return ts,rs
    else:
        return t
    
def cont_sim1D_RDT(params,r0, trajectory = False):
    """
    Generate one trajectory for rotationally invariant diffusive particle using spherical coordinates using the Euler-Maruyama algorithm. Follows the SDE in Eq. 5 of the main text.

    Parameters
    ----------
    params : list
        List of input parameters, in order:
            D: Diffusion constant
            v: Drift constant
            WR: Reset rate, 1/tau
            R1: Position of the reflecting boundary
            R0: Position of the absorbing boundary.
    r0 : float
        Initial therapy efficacy.
    trajectory : boolean, optional
        If true, returns the trajectory of the process. The default is False.
    Returns
    -------
    if history = True:
        t : array
            Time array of the trajectory.
        r : array
            Position array of the trajectory (in radial distances).
    else:
        t : float
            Time of the trajectory at termination (RDT)
    """
    
    D,v,WR,R1,R0=params
    dt=0.01/np.sqrt(D)
    sqdh2=np.sqrt(2.0*D*dt)

    #Initial conditions
    r=r0
    t=0
    if trajectory:
        rs=[r0];ts=[t]

    #Updates reset time
    if WR > 0:
        tR=-1/WR*np.log(np.random.uniform()) #Time for first reset
    else:
        tR = np.inf
        
    clock=0 # Dummy variable to indicate if I reached reset time (tR)

    while r > R0:
        t+=dt
        clock+=dt
        if(clock>tR):
            tR=-1/WR*np.log(np.random.uniform()) # Time for next reset            
            clock=0
            r=r0
        else:
            dr=v*dt+sqdh2*np.random.normal()
            rn=r+dr
            if(r>R1):
                r=2*R1-rn
            else:
                r=rn
        if trajectory:
            rs.append(r);ts.append(t)
        
    if trajectory:
        return ts,rs
    else:
        return t