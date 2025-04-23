import numpy as np
import scipy.sparse as sp
import itertools as ite

def vDtokpm(v,D,N):
    """
    Conversion of the drift and diffusion constnat to p and q jump rates in the discrete model, 
    using the relation from the Kramers-Moyal expansion, Eq. C4 in the main text

    Parameters
    ----------
    v : float
        Drift constant.
    D : float
        Diffusion constant.
    N : integer
        Number of lattice states.

    Raises
    ------
    ValueError
        If the resulting kp and km are negative, the expansion did not work.

    Returns
    -------
    kp : float
        Transition rate towards higher eta. Denoted as "p" in the main text.
    km : float
        Transition rate towards lower eta. Denoted as "q" in the main text..

    """

    kp = (1/2)*((2*D*(N**2)) + (N*v))
    km = (1/2)*((2*D*(N**2)) - (N*v))
    
    if kp<0 or km<0:
        raise ValueError("Invalid v and D")
    
    return kp, km

def state_to_ind(state, N):
    """
    Convert lattice state coordinates on the lattice to matrix indices

    Parameters
    ----------
    state : integer tuple
        State coordinate.
    N : integer
        Number of lattice states.

    Returns
    -------
    ind : integer
        Matrix index.

    """
    i = state[0]
    j = state[1]
    ind = (i)*N+j
    return ind

def ind_to_state(ind, N):
    """
    Convert matrix indices to lattice state coordinates

    Parameters
    ----------
    ind : integer
        Matrix index.
    N : integer
        Number of lattice states.

    Returns
    -------
    state : integer tuple
        State coordinate.

    """
    i = np.floor(ind/N)
    j = ind-(i*N)
    return int(i), int(j)

def Mmat(vx, vy, Dx, Dy, N):
    """
    Transition matrix of a complete lattice graph, as discussed in Appendix C2

    Parameters
    ----------
    vx : float
        Drift constant of therapy 1.
    vy : float
        Drift constant of therapy 2.
    Dx : float
        Diffusion constant of therapy 1.
    Dy : float
        Diffusion constant of therapy 2.
    N : integer
        Number of lattice states.

    Returns
    -------
    M : Array
        Transition matrix of a complete lattice graph.

    """

    kx = vDtokpm(vx, Dx, N)
    ky = vDtokpm(vy, Dy, N)
    
    kxp = kx[0]
    kxm = kx[1]
    
    kyp = ky[0]
    kym = ky[1]    

    #Generate transitions along the x-axis
    kxplist = list(ite.repeat(kxp, N-1))
    kxmlist = list(ite.repeat(kxm, N-1))
    kxdiag = sp.diags([kxplist, kxmlist], [-1,1])
    kxdiaglist = list(ite.repeat(kxdiag, N))
    kxmat = sp.block_diag(kxdiaglist)
    
    zeromat = sp.csr_matrix((0,N))
    
    #Generate upward transitions along the y-axis
    kyplist = list(ite.repeat(kyp, N))
    kypdiag = sp.diags(kyplist)
    kypdiaglist = list(ite.repeat(kypdiag, N-1))
    kypmatlist = [zeromat.T] + kypdiaglist + [zeromat]
    kypmat = sp.block_diag(kypmatlist)
    
    #Generate downward transitions along the y-axis
    kymlist = list(ite.repeat(kym, N))
    kymdiag = sp.diags(kymlist)
    kymdiaglist = list(ite.repeat(kymdiag, N-1))
    kymmatlist = [zeromat] + kymdiaglist + [zeromat.T]
    kymmat = sp.block_diag(kymmatlist)
    
    #Generate full matrix block for outward transitions
    kmat = (kxmat + kypmat + kymmat)
    
    #Generate diagonal elements for rightward transitions along the x-axis
    kxplist1 = kxplist + [0]
    kxpd = sp.diags(kxplist1)
    kxpdiaglist1 = list(ite.repeat(kxpd, N))
    kxpdiag1 = sp.block_diag(kxpdiaglist1)
    
    #Generate diagonal elements for leftward transitions along the x-axis
    kxmlist1 = [0] + kxmlist
    kxmd = sp.diags(kxmlist1)
    kxmdiaglist1 = list(ite.repeat(kxmd, N))
    kxmdiag1 = sp.block_diag(kxmdiaglist1)
    
    #Generate diagonal elements for downward transitions along the x-axis
    kypd = sp.diags(kyplist)
    kypdiaglist1 = list(ite.repeat(kypd, N-1)) + [sp.csr_matrix((N,N))]
    kypdiag1 = sp.block_diag(kypdiaglist1)
    
    #Generate diagonal elements for upward transitions along the x-axis
    kymd = sp.diags(kymlist)
    kymdiaglist1 = [sp.csr_matrix((N,N))] + list(ite.repeat(kymd, N-1))
    kymdiag1 = sp.block_diag(kymdiaglist1)
    
    #Generate full matrix diagonal for inward transitions
    kdiag = kxpdiag1 + kxmdiag1 + kypdiag1 + kymdiag1
    
    M = (kmat - kdiag)
    
    return M

def Rmat(rx, ry, startposx, startposy, N):
    """
    Transition matrix of therapy switching, as discussed in Appendix C2

    Parameters
    ----------
    rx : float
        Inverse of therapy switching rate of therapy 1, 1/tau1.
    ry : float
        Inverse of therapy switching rate of therapy 2, 1/tau2.
    startposx : integer
        Initial lattice state of therapy 1.
    startposy : integer
        Initial lattice state of therapy 2.
    N : integer
        Number of lattice states.

    Returns
    -------
    R : Array
        Transition matrix of therapy switching.

    """
    
    #Generate outward reset transitions along the x-axis
    rxlist = list(ite.repeat(rx, N))
    rxlist[startposx] = 0
    rxmat = sp.lil_matrix((N,N))
    rxmat[startposx] = rxlist
    rxmatlist = list(ite.repeat(rxmat, N))
    rxbigmat = sp.block_diag(rxmatlist)
    
    #Generate inward reset transitions along x-axis
    rxdiag = sp.diags(rxlist)
    rxdiaglist = list(ite.repeat(rxdiag, N))
    rxbigdiag = sp.block_diag(rxdiaglist)
    
    #Generate matrix block for resets along x-axis
    R_X = (rxbigmat-rxbigdiag)
    
    #Generate outward reset transitions along the y-axis
    rylist = list(ite.repeat(ry, N))
    rymat = sp.diags(rylist)
    rymatlist = list(ite.repeat(rymat,N))
    rymatlist[startposy] = sp.lil_matrix((N,N))
    rymatblock = sp.hstack(rymatlist)
    rymatblocklist = list(ite.repeat(sp.lil_matrix((N, N**2)), N))
    rymatblocklist[startposy] = rymatblock
    rybigmat = sp.vstack(rymatblocklist)
    
    #Generate inward reset transitions along y-axis
    rydiaglist = rymatlist.copy()
    rydiaglist[startposy] = sp.lil_matrix((N,N))
    rybigdiag = sp.block_diag(rydiaglist)
    
    #Generate matrix block for resets along y-axis
    R_Y = (rybigmat-rybigdiag)
    
    #Generate final transition matrix for resetting
    R = (R_X + R_Y)
    
    return R

def Cmat(vx, vy, Dx, Dy, rx, ry, startposx, startposy, N):
    """
    Transition matrix of partial absorbing boundaries, as discussed in Appendix C2

    Parameters
    ----------
    vx : float
        Drift constant of therapy 1.
    vy : float
        Drift constant of therapy 2.
    Dx : float
        Diffusion constant of therapy 1.
    Dy : float
        Diffusion constant of therapy 2.
    rx : float
        Inverse of therapy switching rate of therapy 1, 1/tau1.
    ry : float
        Inverse of therapy switching rate of therapy 2, 1/tau2.
    startposx : integer
        Initial lattice state of therapy 1.
    startposy : integer
        Initial lattice state of therapy 2.
    N : integer
        Number of lattice states.

    Returns
    -------
    C : Array
        Transition matrix of absorption.

    """

    kx = vDtokpm(vx, Dx, N)
    ky = vDtokpm(vy, Dy, N)
    
    kxp = kx[0]
    
    kyp = ky[0]

    C = sp.lil_matrix((N**2,N**2))
    
    #Recreate matrix transitions along the axis
    for i in np.arange(0,N):
        absindx1 = state_to_ind((i,1),N)
        absindx2 = state_to_ind((i,0),N)
        C[absindx1, absindx2] = C[absindx1, absindx2]+kxp
        C[absindx2, absindx2] = C[absindx2, absindx2]-kxp
        
        absindy1 = state_to_ind((0,i),N)
        absindy2 = state_to_ind((1,i),N)
        C[absindy1, absindy1] = C[absindy1, absindy1]-kyp
        C[absindy2, absindy1] = C[absindy2, absindy1]+kyp
    
    #Recreate reset transitions along the axis
    C[0,0] = C[0,0]-rx-ry
    resindx = state_to_ind((0,startposx),N)
    resindy = state_to_ind((startposy,0),N)
    C[resindx,0] = C[resindx,0]+rx
    C[resindy,0] = C[resindy,0]+ry
    
    return C

def Wmat(vx, vy, Dx, Dy, rx, ry, startposx, startposy, N):
    """
    Complete transition matrix, complete absorbing states need to be removed first

    Parameters
    ----------
    vx : float
        Drift constant of therapy 1.
    vy : float
        Drift constant of therapy 2.
    Dx : float
        Diffusion constant of therapy 1.
    Dy : float
        Diffusion constant of therapy 2.
    rx : float
        Inverse of therapy switching rate of therapy 1, 1/tau1.
    ry : float
        Inverse of therapy switching rate of therapy 2, 1/tau2.
    startposx : integer
        Initial lattice state of therapy 1.
    startposy : integer
        Initial lattice state of therapy 2.
    N : integer
        Number of lattice states.

    Returns
    -------
    W : Array
        Transition matrix.

    """
    M = Mmat(vx, vy, Dx, Dy, N)
    R = Rmat(rx, ry, startposx, startposy, N)
    C = Cmat(vx, vy, Dx, Dy, rx, ry, startposx, startposy, N)
    return sp.lil_matrix((M+R-C))

def WSbound(vx, vy, Dx, Dy, rx, ry, startposx, startposy, rho, N):
    """
    Components for computing the mean RDT for the uncoupled discrete model

    Parameters
    ----------
    vx : float
        Drift constant of therapy 1.
    vy : float
        Drift constant of therapy 2.
    Dx : float
        Diffusion constant of therapy 1.
    Dy : float
        Diffusion constant of therapy 2.
    rx : float
        Inverse of therapy switching rate of therapy 1, 1/tau1.
    ry : float
        Inverse of therapy switching rate of therapy 2, 1/tau2.
    startposx : integer
        Initial lattice state of therapy 1.
    startposy : integer
        Initial lattice state of therapy 2.
    N : integer
        Number of lattice states.

    Returns
    -------
    W : Array
        Transition matrix of the model.
    S : Array
        Transition matrix of the model with absorbing states removed/Survival matrix.
    delstate : list
        List of states falling under the absorbing boundary.
    delind : list
        List of matrix indices falling under the absorbing boundary.
    bdstate : list
        List of states along the absorbing boundary.
    bdind : list
        List of matrix indices along the absorbing boundary.

    """
    W = Wmat(vx, vy, Dx, Dy, rx, ry, startposx, startposy, N)
    statelist = [ind_to_state(i, N) for i in np.arange(0,N**2)]
    
    #Generate list of states that fall under the complete absorbing boundary to be deleted
    delstate = [s for s in statelist if np.floor(np.sqrt(s[0]**2 + s[1]**2)) <= rho]

    #Remove transitions towards states that fall under the complete absorbing boundary
    delind = [state_to_ind(s, N) for s in delstate]
    for d in delind:
        for i in np.arange(0,N**2):
            W[i,d] = 0
            W[d,i] = 0
    
    #Retrieve the states adjacent to the deleted states, to be designated as absorbing states
    bdx = [(sx[0]+1, sx[1]) for sx in delstate if sx[0]+1<N]
    bdy = [(sy[0], sy[1]+1) for sy in delstate if sy[1]+1<N]
    bdstate = [s for s in list(set(bdx + bdy)) if s not in delstate]
    
    #Remove outward transitions from absorbing states
    bdind = [state_to_ind(s, N) for s in bdstate]
    for b in bdind:
        for i in np.arange(0, N**2):
            W[i,b] = 0
            
    #Generate the survival matrix, as discussed in Harunari et. al. (2022) https://doi.org/10.1103/PhysRevX.12.041026
    S = W.copy()
    for a in bdind:
        for i in np.arange(0, N**2):
            S[a,i] = 0
    return W, S, delstate, delind, bdstate, bdind

def disc_ana_MRDT(params,startposx, startposy):
    """
    Analytical mean RDT of the uncoupled discrete model, given by Harunari et. al. (2022) https://doi.org/10.1103/PhysRevX.12.041026 and Eq. 9 of the main text

    Parameters
    ----------
    params : list
        List of input parameters, in order:
            N: Number of lattice states
            rho: Position of the absorbing boundary
            rx: Reset rate of therapy 1, 1/tau1
            ry: Reset rate of therapy 2, 1/tau2
            vx: Drift constant of therapy 1
            vy: Drift constant of therapy 2
            Dx: Diffusion constant of therapy 1
            Dy: Diffusion constant of therapy 2.
    startposx : integer
        State position of therapy 1.
    startposy : integer
        State position of therapy 2.

    Returns
    -------
    mrdt : float
        Mean resistance development time of the uncoupled discrete model.

    """
    N = params[0]
    rho = params[1]
    
    rx = params[2]
    ry = params[3]
    
    vx = params[4]
    vy = params[5]
    
    Dx = params[6]
    Dy = params[7]

    W, S, delstate, delind, bdstate, bdind = WSbound(vx, vy, Dx, Dy, rx, ry, startposx, startposy, rho, N)
    
    #Compute for the 2nd power of the inverse of the survival matrix, removes the empty rows and columns of the survival matrix
    Sarr = S.toarray()
    Sarr2 = Sarr[np.ix_(Sarr.any(1), Sarr.any(1))]
    Sinv = np.linalg.inv(Sarr2)
    S2 = np.linalg.matrix_power(Sinv,2)

    tot_del = np.sort(delind + bdind)
    
    #Take the state coordinates and indices of the states adjacent to the absorbing state
    lx = [(sx[0]+1, sx[1]) for sx in bdstate if sx[0]+1<N]
    ly = [(sy[0], sy[1]+1) for sy in bdstate if sy[1]+1<N]
    lstate = [s for s in list(set(lx + ly)) if s not in bdstate]
    lind = [state_to_ind(s, N) for s in lstate]
    
    #These will be used to compensate for the index so that the indices of the transition and the survival matrix still match
    startind = state_to_ind((startposx, startposy),N)
    s_off = len(tot_del[np.where(tot_del<startind)])
    
    mrdt = 0
    #Taking the product of the matrix elements of the transitions to the absorbing state
    prodvals = list(ite.product(lind,bdind))
    for p in np.arange(0, len(prodvals)):
        l = prodvals[p][0]
        b = prodvals[p][1]
        
        if W[b,l] != 0:
            l_off = len(tot_del[np.where(tot_del<l)])
            mrdt += W[b,l]*S2[l-l_off,startind-s_off]
    
    return mrdt

def simmat(vx, vy, Dx, Dy, rx, ry, startposx, startposy, N):
    """
    Modified transition matrix for simulations

    Parameters
    ----------
    vx : float
        Drift constant of therapy 1.
    vy : float
        Drift constant of therapy 2.
    Dx : float
        Diffusion constant of therapy 1.
    Dy : float
        Diffusion constant of therapy 2.
    rx : float
        Inverse of therapy switching rate of therapy 1, 1/tau1.
    ry : float
        Inverse of therapy switching rate of therapy 2, 1/tau2.
    startposx : integer
        Initial lattice state of therapy 1.
    startposy : integer
        Initial lattice state of therapy 2.
    N : integer
        Number of lattice states.

    Returns
    -------
    Array
        Transition matrix for simulations.

    """

    W = Wmat(vx, vy, Dx, Dy, 0, 0, startposx, startposy, N)
    W.setdiag(0)
    
    #Include two additional rows to the unmodified transition matrix that both contains the switching rates
    Rlist1 = list(ite.repeat(rx,N**2))
    Rlist2 = list(ite.repeat(ry,N**2))
    Rlist1[0] = 0
    Rlist1[startposx] = 0
    Rlist2[0] = 0
    Rlist2[startposy] = 0
    
    M = sp.vstack([W, sp.csr_matrix(Rlist1), sp.csr_matrix(Rlist2)])
    
    return sp.csr_matrix(M)

def disc_sim_RDT(params, startposx, startposy, res_type = "unlimited", lim = 0, cost = np.inf, trajectory = False):
    """
    Generate one trajectory for a random walk on a lattice using the Gillespie algorithm. Follows the master equation and construction of the transition matrix in Eq. 7 of the main text

    Parameters
    ----------
    params : list
        List of input parameters, in order:
            N: Number of lattice states
            rho: Position of the absorbing boundary
            rx: Reset rate of therapy 1, 1/tau1
            ry: Reset rate of therapy 2, 1/tau2
            vx: Drift constant of therapy 1
            vy: Drift constant of therapy 2
            Dx: Diffusion constant of therapy 1
            Dy: Diffusion constant of therapy 2.
    startposx : integer
        State position of therapy 1.
    startposy : integer
        State position of therapy 2.
    res_type : "unlimited", "limited", "costed", optional
        "unlimited" : No constraints in therapy switching
        "limited" : Explicit limit on therapy switching
        "costed" : Subsequent switches impose a cost, following Eq. 12
        The default is "unlimited"
    lim : integer, optional
        Used when res_type = "limited". Allowed limit of therapy switching.
        The default is 0
    cost : float, optional
        Used when res_type = "costed". Cost parameter for therapy switching. The default is np.inf. 
    trajectory : TYPE, optional
        If true, returns the trajectory of the process. The default is False.

    Returns
    -------
    if trajectory = True:
        posvals : array
            Position array of the trajectory.
        tvals : array
            Time array of the trajectory.
    else:
        t : float
            Time of the trajectory at termination (RDT)

    """
    
    N, rho, rx, ry, vx, vy, Dx, Dy = params
    M = simmat(vx, vy, Dx, Dy, rx, ry, startposx, startposy, N)
    
    statelist = [ind_to_state(i, N) for i in np.arange(0,N**2)]
    startpos = (startposx, startposy)
    
    #Initialize the position and convert to a matrix index on the transition matrix
    current_pos = startpos
    current_index = state_to_ind(startpos, N)
    
    current_t = 0
    
    if trajectory:
        tvals = []
        posvals = []
    
    gamma = 0 
    while True:
        #Take the column of rates corresponding to the current position
        Mrates = M[:,current_index].transpose()
        
        if res_type == "limited":
            #For limited switching
            #If the limit has been reached, remove all switching rates from the transition matrix
            if lim <= 0:
                Mrates = sp.lil_matrix(Mrates)
                Mrates[:,-2] = 0
                Mrates[:,-1] = 0       
                Mrates = sp.csr_matrix(Mrates)
        
        #Following Gillespie algorithm, compute for the next time increment tau using the rates
        a0 = sp.csr_matrix.sum(Mrates)
        if a0 == 0:
            print(current_index)
        tau = np.random.exponential(scale = 1/a0)
        
        #Choose the next transition Rpick
        rand = np.random.uniform(0,1)        
        prod_rand_rate = a0*rand
        Rpick = (prod_rand_rate <= np.cumsum(Mrates.toarray()))
        R = np.where(Rpick == True)[0][0]
        
        if res_type == "limited":
            #For limited resetting
            #If the reset transitions are chosen, return the position to the original position on its corresponding axis
            if R == len(Rpick)-2 and lim != 0:
                current_pos = (startpos[0], current_pos[1])
                current_index = state_to_ind(current_pos, N)
                #Decrement the limit at every reset
                lim -= 1
    
            elif R == len(Rpick)-1 and lim != 0:
                current_pos = (current_pos[0], startpos[1])
                current_index = state_to_ind(current_pos, N)
                #Decrement the limit at every reset
                lim -= 1
    
            else:
                if np.sqrt(statelist[R][0]**2 + statelist[R][1]**2) > rho:
                    #Normal transition based on Rpick
                    current_pos = statelist[R]
                    current_index = state_to_ind(current_pos,N)
                else:
                    #Absorbing boundary condition
                    break
        elif res_type == "costed": 
            #For limited resetting
            #If the reset transitions are chosen, return the position to the original position on its corresponding axis
            if R == len(Rpick)-2:
                current_pos = (startpos[0], current_pos[1])
                current_index = state_to_ind(current_pos, N)
                
                #Compute for the new costed reset rate after each reset
                gamma += 1
                rxcost = rx*(np.exp(-(365*cost)*rx*(gamma)))
                rycost = ry*(np.exp(-(365*cost)*ry*(gamma)))
                M = simmat(vx, vy, Dx, Dy, rxcost, rycost, startposx, startposy, N)
            
            elif R == len(Rpick)-1:
                current_pos = (current_pos[0], startpos[1])
                current_index = state_to_ind(current_pos, N)
                
                #Compute for the new costed reset rate after each reset
                gamma += 1
                rxcost = rx*(np.exp(-(365*cost)*rx*(gamma)))
                rycost = ry*(np.exp(-(365*cost)*ry*(gamma)))
                M = simmat(vx, vy, Dx, Dy, rxcost, rycost, startposx, startposy, N)
            else:
                if np.sqrt(statelist[R][0]**2 + statelist[R][1]**2) > rho:
                    #Normal transition based on Rpick
                    current_pos = statelist[R]
                    current_index = state_to_ind(current_pos,N)
                else:
                    #Absorbing boundary condition
                    break
        elif res_type == "unlimited":
            #For standard resetting
            #If the reset transitions are chosen, return the position to the original position on its corresponding axis
            if R == len(Rpick)-2:
                current_pos = (startpos[0], current_pos[1])
                current_index = state_to_ind(current_pos, N)
    
            elif R == len(Rpick)-1:
                current_pos = (current_pos[0], startpos[1])
                current_index = state_to_ind(current_pos, N)
    
            else:
                if np.sqrt(statelist[R][0]**2 + statelist[R][1]**2) > rho:
                    #Normal transition based on Rpick
                    current_pos = statelist[R]
                    current_index = state_to_ind(current_pos,N)
                else:
                    #Absorbing boundary condition
                    break
        
        #Increment the time by the computed increment tau
        current_t  = current_t + tau
        
        if trajectory:
            #Record the position if trajectoy = True
            tvals.append(current_t)
            posvals.append(current_pos)
    
    if trajectory:
        return posvals, tvals
    else:
        return current_t
    
def disc_sim1D_RDT(params, x0, trajectory = False):
    """
    Generates a simulated trajectory of a random walk on a chain of states with resetting, solved using the Gillespie algorithm

    Parameters
    ----------
    params : list
        v : float. Drift constant.
        D : float. Diffusion constant.
        M : integer. Number of states in the discrete model
        r : float. Resetting rate, inverse of therapy switching rate
    x0 : float.
        Initial therapy efficacy.
    trajectory : boolean, optional
        If true, return the trajectory. The default is False.

    Returns
    -------
    if trajectory == True
        xv : list.
            Position values of the random walk.
        tv : list.
            Time values of the random walk.
    else:
        t : float.
            First passage time/resistance development time

    """
    #Initialize parameters
    v,D,M,r = params
    k_plus, k_minus = vDtokpm(v,D,M) #Convert drift and diffusion constant to jump rates

    #Initialize time and position
    x = x0
    t = 0
    
    if trajectory:
        #If recording trajectory, initialize arrays
        xv = [x0]
        tv = [t]
    
    #Propensity functions
    aj = [k_plus,
          k_minus,
          r]
    a0 = sum(aj)
    
    while x>0:
        
        #Compute the next time increment tau using the sum of the propensity functions
        tau = np.random.exponential(scale = 1/a0)
        t  = t + tau
        
        #Choose the next reaction
        rand = np.random.uniform(0,1)        
        prod_rand_rate = a0*rand
        Rpick = (prod_rand_rate <= np.cumsum(aj))
        R = np.where(Rpick == True)[0][0]
        
        if R == 0:
            #Move to increase x
            if x + (1/M) >= 1:
                x = x
            else:
                x = x + (1/M)
        elif R == 1:
            #Move to decrease x
            if x - (1/M) <= 0:
                x = 0
            else:
                x = x - (1/M)
        elif R == 2:
            #Reset
            x = x0
        
        if trajectory:
            xv.append(x)
            tv.append(t)
            
    if trajectory:
        return tv, xv
    else:
        return t