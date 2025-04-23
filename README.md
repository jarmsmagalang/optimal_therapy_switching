# Optimal switching strategies in multi-drug therapies for chronic diseases --- Figures
**Juan Magalang, Javier Aguilar, Jose Perico Esguerra, Edgar Roldan, Daniel Sanchez-Taltavull**

This repository is a collection of scripts used in generating the figures in the main text[^1]. It also contains the numerical implementations of the mathematical expressions found in the main text, which will be described in this documentation. These functions may be found in the `src` folder.

This readme is currently a **work in progress**, however the documentation of each main function is included as docstrings and can be accessed by using `help(name_of_function)`.

# Fixed Points of the Healthy Cell Population
Filename: `src/H_fixedpt.py`

The main function of this package is the `H_fixedpt` function, which gives the fixed point of the healthy cell population (Eqs. A1 and A2 in the main text), given $N_T = 2$, with the following syntax:
```
H_fixedpt(eta1, eta2)
```
Here is a description of each argument:
- `eta1` : float. Therapy efficacy of therapy 1.
- `eta2` : float. Therapy efficacy of therapy 2.

Sample usage:

```
from src.H_fixedpt import H_fixedpt

eta1 = 0.4
eta2 = 0.3
fixedpt = H_fixedpt(eta1, eta2)
```
Sample output:
```
>>> fixedpt
476925.4218246925
```

# Coupled Continuous Model
Filename: `src/coupled_continuous.py`

This script contains various functions for the coupled continuous model featured in Section IIA of the main text.

### Mean RDT -- `cont_ana_MRDT`
Calculates the mean RDT of the coupled continuous model, given by Eq. 6 of the main text.
```
cont_ana_MRDT(params,r)
```
Parameters:
- `params` : list. List of input parameters, in order:
	* `D`: Diffusion constant
	* `v` : Drift constant
	* `WR`: Reset rate, $1/\tau$
	* `d`: Number of therapies, $N_T$
	* `R1`: Position of the reflecting boundary
	* `R0`: Position of the absorbing boundary
- `r` : float. Initial therapy efficacy.

```
from src.coupled_continuous import cont_ana_MRDT

D = 10**(-4)
v = -8*(10**(-5))
tau = 3
WR = 1/(tau*365)
d = 2
R1 = 1.0
R0 = 0.4

params = [D, v, WR, d, R1, R0]
mrdt = cont_ana_MRDT(params, 0.8)
```
Sample output:
```
>>> mrdt
1926.7093956106062
```

### Simulated RDT --  `cont_sim_MRDT`

Generates a simulated value of the RDT or a trajectory of the coupled continuous model, with or without therapy switching. Simulations generated using the Euler-Maruyama algorithm.

```
cont_sim_MRDT(params,r0, res_type = "unlimited", AT = 0, trajectory = False)
```
Parameters:
- `params` : list. List of input parameters, in order:
	* `D`: Diffusion constant
	* `v` : Drift constant
	* `WR`: Reset rate, $1/\tau$
	* `d`: Number of therapies, $N_T$
	* `R1`: Position of the reflecting boundary
	* `R0`: Position of the absorbing boundary
- `r0` : float. Initial therapy efficacy.
- `res_type` : string, optional. Therapy switching type. The default is `"unlimited"`.
	 * `"unlimited"` : No constraints in therapy switching
     * `"limited"` : Explicit limit on therapy switching
	 * `"costed"` : Subsequent switches impose a cost, following Eq. 12 with $c = 10^{(d-1)}$
- `AT` : integer, optional. Used when `res_type = "limited"`. Allowed limit of therapy switching, described in Eq. 11. The default is 0.
- `trajectory` : boolean, optional. If true, returns the trajectory of the process. The default is False.

```
from src.coupled_continuous import cont_sim_MRDT

D = 10**(-4)
v = -8*(10**(-5))
tau = 3
WR = 1/(tau*365)
d = 2
R1 = 1.0
R0 = 0.4

params = [D, v, WR, d, R1, R0]
rdt = cont_sim_RDT(params, 0.8)
tval, xval = cont_sim_RDT(params, 0.8, trajectory = True, res_type = "limited", AT = 12)
```
Sample output:
```
>>> rdt
446.0
>>> tval
[0, 1.0, 2.0, ...]
>>> xval
[0.8, 0.8092825171433242, 0.7897353534596738, 0.7749005475672365, ...]
```


[^1]: [Optimal switching strategies in multi-drug therapies for chronic diseases](https://arxiv.org/abs/2411.16362)
> Written with [StackEdit](https://stackedit.io/).
