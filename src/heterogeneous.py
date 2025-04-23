import scipy.sparse as sp
import itertools as ite
import numpy as np
import scipy.integrate as integrate
import scipy.stats as spst
import copy as cp
import warnings 

def vjmat(N):
    """
    Generates a matrix of state change vectors.
    Rows correspond to the populations of latently-infected and actively-infecting cell strains.
    Columns correspond to the state change vector for each propensity function ordered as follows:
        - Infection to generate Li, no mutation
        - Infection to generate Li, mutate to increase state
        - Infection to generate Li, mutate to decrease state
        - Infection to generate Ii, no mutation
        - Infection to generate Ii, mutate to increase state
        - Infection to generate Ii, mutate to decrease state
        - Proliferation of Li cells
        - Carrying capacity over all combinations of Li cells
        - Death of Li cells
        - Death of Ii cells

    Parameters
    ----------
    N : integer
        Number of strains for both L and I.

    Returns
    -------
    vj : sparse matrix
        Matrix of state change vectors

    """
    #Generate state change vectors for H + I -> L + I reaction for each mutation strain
    HL = sp.diags(list(ite.repeat(1,N))) #No mutation
    # HLp = sp.diags([0]+list(ite.repeat(1,N-2)), -1) #Mutation, increase state
    # HLm = sp.diags(list(ite.repeat(1,N-1)),1) #Mutation, decrease state
    
    #Generate state change vectors for H + I -> I + I reaction for each mutation strain
    HI = sp.diags(list(ite.repeat(1,N))) #No mutation
    # HIp = sp.diags([0]+list(ite.repeat(1,N-2)), -1) #Mutation, increase state
    # HIm = sp.diags(list(ite.repeat(1,N-1)),1) #Mutation, decrease state
    
    #Generate state change vectors for L -> L + L reaction for each mutation strain
    pL = sp.diags(list(ite.repeat(1,N)))
    pLp = sp.diags([0]+list(ite.repeat(1,N-2)), -1)
    pLm = sp.diags(list(ite.repeat(1,N-1)),1)
    
    #Generate state change vectors for L + L -> 0 reaction for all possible combinations of the mutation strains in L
    cLarr = []
    zerarr = []
    for n in np.flip(np.arange(1,N+1)):
        clist = list(ite.repeat(-1,n))
        cL1 = sp.diags(clist)
        cL2 = sp.csr_matrix((clist, (list(ite.repeat(0,n)), np.arange(0,n))), shape = (n,n))
        cL = cL1+cL2
        
        cLzero = sp.csr_matrix((N-cL.shape[0], cL.shape[1]), dtype = np.float64())
        cL = sp.vstack([cLzero, cL])
    
        cLarr.append(cL)
        zerarr.append(sp.csr_matrix((N, cL.shape[1])))
    
    #Generate state change vectors for L -> I reaction for each mutation strain
    aL_L = sp.diags(list(ite.repeat(-1,N)))
    aL_I = sp.diags(list(ite.repeat(1,N)))
    
    #Generate state change vectors for L -> 0 reaction for each mutation strain
    dL = sp.diags(list(ite.repeat(-1,N)))
    #Generate state change vectors for I -> 0 reaction for each mutation strain
    dI = sp.diags(list(ite.repeat(-1,N)))
    
    #Filler matrix
    zer = sp.csr_matrix((N,N))
    
    #Generate matrix of state change vectors for all L
    Llist = list(ite.chain.from_iterable([[HL], 
                                          # [HLp], 
                                          # [HLm], 
                                          [zer], 
                                          # [zer], 
                                          # [zer], 
                                          [pL], 
                                          [pLp], 
                                          [pLm], 
                                           cLarr, 
                                          [aL_L], 
                                          [dL], 
                                          [zer],]))
    Lstate = sp.hstack(Llist)
    
    #Generate matrix of state change vectors for all I
    Ilist = list(ite.chain.from_iterable([[zer], 
                                          # [zer], 
                                          # [zer], 
                                          [HI], 
                                          # [HIp], 
                                          # [HIm], 
                                          [zer], 
                                          [zer], 
                                          [zer], 
                                           zerarr, 
                                          [aL_I], 
                                          [zer], 
                                          [dI],]))
    Istate = sp.hstack(Ilist)
    
    #Stack L and I state change vector matrices vertically
    vj = sp.vstack([Lstate, Istate])
    return vj

def ajvec(N, Svals, current_H,
          alpha, lambda_H, p, beta, epsilon, a_L, lambda_L, lambda_I, K, mu, sigma):
    """
    Generates a vector of propensity functions, following the mathematical details in Appendix D

    Parameters
    ----------
    N : integer
        Number of mutation states in L and I.
    Svals : array
        Population counts of all Li and Ii.
    current_H : float
        Population counts of healthy susceptible cells.
    alpha : float
        Recruitment constant of H.
    lambda_H : float
        Death constant of H.
    p : float
        Proliferation constant of Li.
    beta : float
        Infection constant.
    epsilon : float
        Probability of infection yielding Li.
    a_L : float
        Activation constant of Li.
    lambda_L : float
        Death constant of Li.
    lambda_I : float
        Death constant of Li.
    K : float
        Carrying capacity for Li.
    mu : float
        Mutation probability.
    sigma : float
        Mutation state transition probability.

    Returns
    -------
    aj : array
        Vector of propensity functions.

    """

    #Retrieve the population counts for the Li and Ii
    Svals2 = cp.deepcopy(Svals)
    Lvals = Svals2[0:N]
    Ivals = Svals2[N:2*N]
    
    #Generate the space for the eta depending on the amount of mutation states
    etavals = np.linspace(0,1,N, endpoint = False)
    
    #Generate propensity functions for H + I -> L + I reaction for each mutation strain
    HLrate = [epsilon*(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(0,N)]
    # HLprate = [0]+[mu*sigma*epsilon*(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(1,N-1)]+[0]
    # HLmrate = [0]+[mu*(1-sigma)*epsilon*(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(1,N-1)] + [mu*epsilon*(1-etavals[-1])*beta*current_H*Ivals[-1]]
    
    #Generate propensity functions for H + I -> I + I reaction for each mutation strain
    HIrate = [(1-epsilon)*(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(0,N)]
    # HIprate = [0]+[mu*sigma*(1-epsilon)*(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(1,N-1)]+[0]
    # HImrate = [0]+[mu*(1-sigma)*(1-epsilon)*(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(1,N-1)] + [mu*(1-epsilon)*(1-etavals[-1])*beta*current_H*Ivals[-1]]
    
    #Generate propensity functions for L -> L + L reaction for each mutation strain
    pLrate = [(1-mu)*p*Lvals[i] for i in np.arange(0,N)]
    pLprate = [0]+[mu*sigma*p*Lvals[i] for i in np.arange(1,N-1)]+[0]
    pLmrate = [0]+[mu*(1-sigma)*p*Lvals[i] for i in np.arange(1,N)]
    
    #Generate propensity functions for L + L -> 0 reaction for all possible combinations of the mutation strains in L
    cross_L = sorted(list(set(ite.combinations(np.arange(0,N),2))) + list([(i,i) for i in np.arange(0,N)]))
    cLrates = []
    for i in cross_L:
        if i[0] == i[1]:
            cLrates.append((p/K)*Lvals[i[0]]*(Lvals[i[1]]-1))
        else:
            cLrates.append((p/K)*Lvals[i[0]]*(Lvals[i[1]]))
    
    #Generate propensity functions for L -> I reaction for each mutation strain
    aLrate = [a_L*Lvals[i] for i in np.arange(0,N)]
    
    #Generate propensity functions for L -> 0 reaction for each mutation strain
    dLrate = [lambda_L*Lvals[i] for i in np.arange(0,N)]
    
    #Generate propensity functions for I -> 0 reaction for each mutation strain
    dIrate = [lambda_I*Ivals[i] for i in np.arange(0,N)]
    
    #Stack the propensity functions into a single vector
    aj1 = [HLrate,
           # HLprate,
           # HLmrate,
           HIrate,
           # HIprate,
           # HImrate,
           pLrate,
           pLprate,
           pLmrate,
           cLrates,
           aLrate,
           dLrate,
           dIrate,]
    aj = np.array(list(ite.chain.from_iterable(aj1)))
    
    return aj

def host_pathogen_params():
    """
    Generates a vector of the input parameters, following Table 1 of the main text.

    Returns
    -------
    H_0 : integer
        Initial healthy cell population.
    L_0 : integer
        Initial latently infected cell population.
    I_0 : integer
        Initial actively infecting cell population.
    alpha : float
        Recruitment constant of H.
    lambda_H : float
        Death constant of H.
    p : float
        Proliferation constant of Li.
    beta : float
        Infection constant.
    epsilon : float
        Probability of infection yielding Li.
    a_L : float
        Activation constant of Li.
    lambda_L : float
        Death constant of Li.
    lambda_I : float
        Death constant of Li.
    K : float
        Carrying capacity for Li.

    """
    H_0 = 599260
    L_0 = 50
    I_0 = 12
    
    alpha = 6000.0
    lambda_H = 0.01
    
    p = 0.2
    
    beta = 5*(10**(-6))
    epsilon = 0.01
    
    a_L = 0.1
    lambda_L = 0.001
    
    lambda_I = 1
     
    K = 100.0

    return (H_0, L_0, I_0, alpha, lambda_H, p, beta, epsilon, a_L, lambda_L, lambda_I, K)

def heterogeneous_RDT(params, trajectory = False, res_type = "unlimited", lim = 0, cost = np.inf, end_time = 1e8, tau = 0.1):
    """
    Generates the trajectory and RDT of the single-cell heterogeneous model, as described in Section V of the main text, using the tau-leaping Gillespie algorithm

    Parameters
    ----------
    params : array
        Array of input parameters:
            N : integer, number of mutation states
            mu : float, mutation probability
            sigma : float, mutation state transition probability
            init_eta : float, initial therapy efficacy
            r : float, inverse of therapy switching rate
    trajectory : boolean, optional
        If true, generates the trajectories for all cell populations. The default is False.
    res_type : string, optional
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
    end_time : float, optional
        Simulation end time. The default is 1e8.
    tau : float, optional
        Tau-leaping algorithm increment. The default is 0.1.

    Returns
    -------
    if trajectory = True:
        ttraj : array
            Time array of the trajectory.
        etatraj : array
            Therapy efficacy array
        Htraj : array
            Healthy cells array
        Ltraj : array
            Latently infected cells array, each row corresponds to a mutation state
        Itraj : array
            Actively infecting cells array, each row corresponds to a mutation state
    else:
        t : float
            Time of the trajectory at termination (RDT)

    """
    #Initialize parameters
    N, mu, sigma, init_eta, r = params
    H_0, L_0, I_0, alpha, lambda_H, p, beta, epsilon, a_L, lambda_L, lambda_I, K = host_pathogen_params()
    
    #Initialize time and cell counts
    current_t = 0
    current_H = H_0
    N0 = int(init_eta*N)
    Lvals = np.zeros(N)
    Ivals = np.zeros(N)
    Lvals[N0] = L_0
    Ivals[N0] = I_0
    Svals = np.concatenate((Lvals, Ivals))
    
    #Initialize state change matrix
    vj = vjmat(N).toarray()
    
    if trajectory:
        #Record all changes in time and cell counts if trajectory = True
        ttraj = [current_t]
        etatraj = [init_eta]
        Htraj = [current_H]
        Ltraj = [Svals[0:N]]
        Itraj = [Svals[N:2*N]]
    
    #Initialize therapy efficacy space
    etavals = np.linspace(0,1,N, endpoint = False)
    
    #Initialize resetting times, depending on the therapy switching rate
    if r > 0:
        reset_time = np.random.exponential(1/r)
    else:
        reset_time = np.inf
    gamma = 0 #For costed resetting, counts how many therapy switches has happened
    
    while True:

        Lvals = Svals[0:N]
        Ivals = Svals[N:2*N]
        
        #Generate propensity function vector given the current cell counts
        aj = ajvec(N, Svals, current_H,
              alpha, lambda_H, p, beta, epsilon, a_L, lambda_L, lambda_I, K, mu, sigma)
        
        #Pick multiple propensity functions from the vector, following the tau-leaping algorithm, and increment cell counts accordingly
        change_vec = spst.poisson.ppf(np.random.random(len(aj)), aj*tau)
        dS = np.sum(vj*change_vec, axis = 1)
        dS_over = cp.deepcopy(dS)
        Svals = Svals + dS_over
        
        #Check if the tau-leaping algorithm caused the cell counts to go negative
        neg_check = np.where(Svals < 0)
        if len(neg_check)>0:
            for neg in neg_check[0]:
                Svals[neg] = 0
        
        #Obtain the healthy cell population by integrating Eq. D1
        H3term = sum([(1-etavals[i])*beta*current_H*Ivals[i] for i in np.arange(0,N)])
        next_H = integrate.quad(lambda t: (alpha - 
                                           lambda_H*current_H - 
                                           H3term), 
                                current_t-tau, current_t)
        current_H = current_H + next_H[0]
        
        #Increment time
        current_t = current_t + tau
        if current_t > reset_time:
            #If the therapy has been switched, take the sums of all Ls and Is
            all_L = np.sum(Lvals)
            all_I = np.sum(Ivals)
            
            #Generate a new cell count array, with the sum of the Ls and Is prior to the switch at the state corresponding to the initial therapy efficacy
            Lvals = np.zeros(N)
            Ivals = np.zeros(N)
            Lvals[N0] = all_L
            Ivals[N0] = all_I
            Svals = np.concatenate((Lvals, Ivals))
            
            Svals = np.zeros(2*N)
            Svals[N0] = all_L
            Svals[2*N0] = all_I
            
            current_t = reset_time
    
            #Generate a new resetting time based on the resetting type
            if res_type == "limited":
                #For limited resetting, decrement the allowed number of therapy switches
                lim -= 1
                if lim <= 0:
                    #Make the resetting time infinite upon consuming all allowed therapy switches
                    reset_time = np.inf
                else:
                    reset_time += np.random.exponential(1/r)
            elif res_type == "costed":
                #For costed resetting, decrease the frequency of resetting based on the number of prior switches, following Eq. 12
                gamma += 1
                rcost = r*np.exp(-cost*r*gamma)
                reset_time += np.random.exponential(1/rcost)
            elif res_type == "unlimited":
                reset_time += np.random.exponential(1/r)
        
        if np.sum(Svals)<=0:
            #Remove outputs if sum of all cell counts are zero
            if trajectory:
                ttraj = [np.nan]
                etatraj = [np.nan]
                Htraj = [np.nan]
                Ltraj = [np.repeat(np.nan, N)]
                Itraj = [np.repeat(np.nan, N)]
            current_t = np.nan
            warnings.warn("Latent and infected cells have all been consumed.")
            break
    
        #Compute for the therapy efficacy, following Eq. 13
        if np.sum(Lvals)>0:
            eta_mean = np.average(etavals, weights = Lvals)
        else:
            eta_mean = np.average(etavals, weights = Ivals)
        
        #Record the time and cell counts if trajectory = True
        if trajectory:
            ttraj.append(current_t)
            etatraj.append(eta_mean)
            Htraj.append(current_H)
            Ltraj.append(Lvals)
            Itraj.append(Ivals)
            
        if current_t > end_time:
            #Remove outputs if simulation time exceeded end_time
            if trajectory:
                ttraj = [np.nan]
                etatraj = [np.nan]
                Htraj = [np.nan]
                Ltraj = [np.repeat(np.nan, N)]
                Itraj = [np.repeat(np.nan, N)]
            current_t = np.nan
            warnings.warn("Simulation time has exceeded the maximum time.")
            break 
        elif eta_mean <= 0:
            #End simulation once eta reaches zero
            if trajectory:
                ttraj.append(current_t)
                etatraj.append(eta_mean)
                Htraj.append(current_H)
                Ltraj.append(Lvals)
                Itraj.append(Ivals)
            break
    
        
    if trajectory:
        return ttraj, etatraj, Htraj, np.array(Ltraj), np.array(Itraj)
    else:
        return current_t
        
