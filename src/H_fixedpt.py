def H_fixedpt(eta1, eta2):
    """
    Fixed points of the healthy cell population for N_T = 2, given by Eqs. A1 and A2 in the text
    
    Parameters
    ----------
    eta1 : float
        Therapy efficacy of therapy 1.
    eta2 : TYPE
        Therapy efficacy of therapy 2.

    Returns
    -------
    Hre : float
        Fixed point of the healthy cell population.

    """
    alpha = 6000
    lambda_H = 0.01

    p = 0.2

    beta = 5*(10**(-6))
    epsilon = 0.01

    a_L = 0.1
    lambda_L = 0.01

    lambda_I = 1

    K = 100
    
    b = (1-eta1)*(1-eta2)*beta
    c = lambda_I*lambda_H-(1-epsilon)*b*alpha
    d = p - a_L - lambda_L
        
    x1 = -p*(lambda_I**2)*(b**2)
    x2 = -(2*p*lambda_I*b*c - d*K*a_L*lambda_I*(b**2))
    x3 = (epsilon*(b**2)*alpha*K*(a_L**2) - p*(c**2) + d*K*a_L*lambda_H*lambda_I*b + d*K*a_L*b*c)
    x4 = (epsilon*b*alpha*K*(a_L**2)*lambda_H + d*K*a_L*lambda_H*c)
    
    if b > 0:
        Q = ((3*x1*x3) - (x2**2))/(9*(x1**2))
        R = ((9*x1*x2*x3) - (27*(x1**2)*x4) - (2*(x2**3)))/(54*(x1**3))
        
        QRsum = int((Q**3)+(R**2))
        
        S = (R + (( QRsum )**(1/2)) )**(1/3)
        T = (R - (( QRsum )**(1/2)) )**(1/3)
            
        I = (S+T) - (x2/(3*x1))
        
        H = (alpha/(lambda_H + b*I))
        Hre = H.real
    else:
        Hre = alpha/lambda_H
    
    return Hre