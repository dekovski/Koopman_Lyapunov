#!/usr/bin/env python3
#
# SSR_code.py | Version 1.0 
#
# This file contains the functions required to generate Figure 2 of the paper
# "Steady State Reduction of generalized Lotka-Volterra systems in the
# microbiome", by Eric Jones and Jean Carlson, published in Physical Review E.
# This code models C. difficile infection (CDI) with the generalized
# Lotka-Volterra (gLV) equations, and demonstrates how this high-dimensional
# microbial state space can be compressed with Steady State Reduction (SSR). In
# total, this program 1) imports the parameters fit by Stein et al. (PLOS Comp.
# Bio. 2013); 2) simulates microbial trajectories of this system; 3) compresses
# the high-dimensional parameters into 2D SSR-reduced parameters; 4) plots the
# reduced SSR-generated trajectories alongside the high-dimensional
# trajectories.  The code is still in a preliminary form. This code is covered
# under GNU GPLv3.
#
# Send questions to Eric Jones at ewj@physics.ucsb.edu
#
###############################################################################

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import pickle

np.set_printoptions(suppress=True, precision=5)

###############################################################################
##### CONTAINER CLASSES THAT CHARACTERIZE GLV AND SSR SYSTEMS
###############################################################################

class Params:
    """ This container class holds the growth rates (rho), interaction
    parameters (K), and antibiotic efficacies (eps)  associated with a gLV
    system. If no input file is provided, the default parameters from Stein et
    al. are used. rho and eps are Nx1, K is NxN. """
    def __init__(s, my_params=None, filename='./data/stein_parameters.csv'):
        # use your own gLV parameters, if desired
        if my_params:
            s.labels, s.rho, s.K = my_params

        else:
            # import "messy" variables and initial conditions from .csv files
            with open(filename, 'r') as f:
                var_data = [line.strip().split(",") for line in f][1:]
            with open('./data/stein_ic.csv','r') as f:
                ic_data = [line.strip().split(",") for line in f]

            # turn "messy" data and ICs into variables
            s.labels, s.rho, s.K, s.eps = parse_data(var_data)
            s.ics = parse_ics(ic_data)
    

    def integrand(s, Y, t, u_params=None):
        """ Integrand for the N-dimensional generalized Lotka-Volterra
        equations, using the parameters contained in s """
        return (np.dot(np.diag(s.rho), Y) + np.dot( np.diag(np.dot(s.K, Y)), Y)
                   + s.u(t, u_params)*np.dot(np.diag(s.eps), Y))

    def u(s, t, u_params):
        """ Returns the concentration of antibiotic currently active """
        try: concentration, duration = u_params
        except TypeError: return 0

        if t < duration:
            return concentration
        else: return 0

    def project_to_2D(s, traj, ssa, ssb):
        """Projects a high-dimensional trajectory traj into a 2D system,
        defined by the origin and steady states ssa and ssb, and returns a
        2-dimensional trajectory, following Eq. S18 of the supplement. Input
        must be of list type (even if that means passing "[point]" instead of
        "point") """
        new_traj = []
        for elem in traj:
            uu = np.dot(ssa, ssa); vv = np.dot(ssb, ssb)
            xu = np.dot(elem, ssa); xv = np.dot(elem, ssb)
            uv = np.dot(ssa, ssb)
            new_traj.append([(xu*vv - xv*uv)/(uu*vv - uv**2),
                             (uu*xv - xu*uv)/(uu*vv - uv**2)])
        new_traj = np.array(new_traj)

        return new_traj

class ssrParams:
    """ This container class holds the parameters for the 2D SSR-reduced
    system, based on the parameters that are passed in p, the steady state ssa,
    and the steady state ssb. The high-dimensional parameters are rho and K;
    the 2D parameters are mu and M """
    def __init__(s, p, ssa, ssb):
        s.mu, s.M = get_SSR_params(p, ssa, ssb)
        s.ssa = ssa
        s.ssb = ssb

    def integrand(s, Y, t):
        """ Integrand for the 2-dimensional generalized Lotka-Volterra
        equations, using the parameters contained in s """
        return np.dot(np.diag(s.mu), Y) + np.dot( np.diag(np.dot(s.M, Y)), Y)

    def get_11_ss(s):
        """ Returns the coexistent steady state (x_a^*, x_b^*) of the 2D gLV
        equations (assuming the equations are not nondimensionalized) """
        xa = - ((-s.M[1][1]*s.mu[0] + s.M[0][1]*s.mu[1]) /
                (s.M[0][1]*s.M[1][0] - s.M[0][0]*s.M[1][1]))
        xb = - ((s.M[1][0]*s.mu[0] - s.M[0][0]*s.mu[1]) /
                (s.M[0][1]*s.M[1][0] - s.M[0][0]*s.M[1][1]))
        return np.array([xa, xb])

###############################################################################
##### HELPER FUNCTIONS
###############################################################################

def parse_data(var_data):
    """ Transforms raw interaction data from the stein_parameters.csv file into
    parameters: labels is the names of each population; mu is the growth rates
    of each population; M[i][j] is the effect of population j on population i;
    eps is the antibiotic susceptibilities of each population"""
    # extract microbe labels, to be placed in legend
    labels = [label.replace("_"," ") for label in var_data[-1] if label.strip()]
    # extract M, mu, and eps from var_data
    str_inter = [elem[1:(1+len(labels))] for elem in var_data][:-1]
    str_gro = [elem[len(labels)+1] for elem in var_data][:-1]
    str_sus = [elem[len(labels)+2] for elem in var_data][:-1]
    float_inter = [[float(value) for value in row] for row in str_inter]
    float_gro = [float(value) for value in str_gro]
    float_sus = [float(value) for value in str_sus]
    # convert to numpy arrays
    M = np.array(float_inter)
    mu = np.array(float_gro)
    eps = np.array(float_sus)
    return labels, mu, M, eps

def parse_ics(ic_data):
    """ Transforms raw initial condition data from the stein_ic.csv file into
    a list of initial conditions (there are 9 experimentally measured initial
    conditions). """
    ic_list_str = [[elem[i] for elem in ic_data][5:-2] for i in \
                    range(1,np.shape(ic_data)[1]) if float(ic_data[3][i])==0]
    ic_list_float = [[float(value) for value in row] for row in ic_list_str]
    ics = np.array(ic_list_float)
    return ics

def solve(p, ic, t_end, interventions={}):
    """ Solves the gLV equations using the parameters given in 'param_list',
    for the scenario specified by 'interventions'. This function also includes
    FMT implementation. u_params = [concentration of dose, duration of dose];
    cd_inoculation = time of CD exposure; transplant_params = [transplant
    composition, transplant size, time of transplantation] """

    # separate 'interventions' parameters into antibiotics (u_params), CD
    # inoculation, and FMT terms
    try: u_params = interventions['u_params']
    except KeyError: u_params = None
    try: cd_inoculation = interventions['CD']
    except KeyError: cd_inoculation = None
    try: transplant_params = interventions['transplant']
    except KeyError: transplant_params = None

    # integrate with no transplant or CD inoculation
    if (not cd_inoculation) and (not transplant_params):
        t = np.linspace(0, t_end, num=101)
        if not u_params:
            y = integrate.odeint(p.integrand, ic, t, atol=1e-12)
        else:
            y = integrate.odeint(p.integrand, ic, t, args=(u_params,), atol=1e-12)
        return t, y

    # integrate with arbitrary transplant
    if transplant_params:
        t_type, t_size, t_time = transplant_params
        if t_time == 0: t_time = 1e-6
        t01 = np.linspace(0, t_time, num=101)
        t12 = np.linspace(t_time, t_end, num=101)
        y01 = integrate.odeint(p.integrand, ic, t01, args=(u_params,))
        # apply transplant:
        new_ic = y01[-1] + np.array([t_size*x for x in t_type])
        y12 = integrate.odeint(p.integrand, new_ic, t12, args=(u_params,))

    # integrate with CD inoculation
    if cd_inoculation:
        t01 = np.linspace(0, cd_inoculation, num=101)
        t12 = np.linspace(cd_inoculation, t_end, num=101)
        y01 = integrate.odeint(p.integrand, ic, t01, args=(u_params,))
        # inoculate w/ CD:
        cd_index = p.labels.index("Clostridium difficile")
        cd_transplant = np.zeros(len(y01[0]))
        cd_transplant[cd_index] = 10**-10
        new_ic = y01[-1] + cd_transplant
        y12 = integrate.odeint(p.integrand, new_ic, t12, args=(u_params,))

    return np.concatenate((t01,t12)), np.vstack((y01,y12))

def get_all_stein_steady_states(p):
    """ Numerically generates all five steady states of the Stein model that
    are reachable from any of the nine experimentally measured initial
    conditions. Steady states are stored in the dictionary "ss_list", with keys
    'A' - 'E'.  Here we obtain each steady state by starting at initial
    conditions 0 or 4, exposing or not exposing the initial condition to a
    small amount of CD, and applying or not applying 1 pulse of antibiotics to
    the system.  For details of how these steady states were "found", see Fig 4
    of Jones and Carlson, PLOS Comp. Bio. 2018.  """

    # 'SS attained': (IC num, if CD exposure, if RX applied)
    ss_conditions = {'A': (0, True, False), 'B': (0, False, False),
                     'C': (4, False, False), 'D': (4, True, True),
                     'E': (4, False, True)}
    ss_list = {}
    for ss in ss_conditions:
        ic_num, if_CD, if_RX = ss_conditions[ss]
        ic = p.ics[ic_num]

        if (not if_CD) and (not if_RX): interventions = {}
        if (if_CD) and (not if_RX): interventions = {'CD': 10}
        if (not if_CD) and (if_RX): interventions = {'u_params': (1, 1)}
        if (if_CD) and (if_RX): interventions = {'u_params': (1, 1), 'CD': 5}

        # solve the gLV ODE for the scenario characterized by 'interventions'
        t, y = solve(p, ic, 10000, interventions)
        ss_list[ss] = np.array([max(yy, 0) for yy in y[-1]])

    return ss_list

def get_SSR_params(p, ssa, ssb):
    """ Given parameters p.rho and p.K, and steady states ssa and ssb, return
    the SSR-generated parameters s.mu and s.M, according to Eqs. 3, A16, and
    A17 of the paper. All parameters are written in terms of the scaled
    variables z_a and z_b as shown in Fig. 2, and as described in Eqs. S20-S22
    of the supplement.  """

    # from Eq 3:
    mu_a = np.dot(np.dot(np.diag(ssa), ssa), p.rho)/(np.linalg.norm(ssa)**2)
    mu_b = np.dot(np.dot(np.diag(ssb), ssb), p.rho)/(np.linalg.norm(ssb)**2)

    # note these are lacking a factor of norm(ssb) or norm(ssa) as given in
    # Eq 3 since we are working in scaled variables (cf. Eqs S20-22)
    M_aa = ( np.dot(np.dot(np.diag(ssa), ssa).T, np.dot(p.K, ssa))
             / (np.linalg.norm(ssa)**2) )
    M_bb = ( np.dot(np.dot(np.diag(ssb), ssb).T, np.dot(p.K, ssb))
             / (np.linalg.norm(ssb)**2) )

    # cross terms from Eq 3 (for orthogonal ssa and ssb):
    # M_ab = ( np.dot(np.dot(np.diag(ssa), ssa).T, np.dot(p.K, ssb))
    #          / (np.linalg.norm(ssa)**2) )
    # M_ba = ( np.dot(np.dot(np.diag(ssb), ssb).T, np.dot(p.K, ssa))
    #          / (np.linalg.norm(ssb)**2) )

    # from Eqs A18 and A19 (complicated cross terms):
    # (ya and yb are as used in the appendix)
    ya = ssa/np.linalg.norm(ssa)
    yb = ssb/np.linalg.norm(ssb)
    numerator = (
        sum([sum([p.K[i][j]*(ya[i]*yb[j] + yb[i]*ya[j])
                  * sum([ya[i]*yb[k]**2 - yb[i]*ya[k]*yb[k]
                         for k in range(len(ssa))])
                  for j in range(len(ssa))])
             for i in range(len(ssa))]) )
    denom = (
        sum([ya[i]**2 for i in range(len(ssa))])
        * sum([yb[i]**2 for i in range(len(ssa))])
        - sum([ya[i]*yb[i] for i in range(len(ssa))])**2 )
    # multiply by norm(ssb) because we are working in scaled variables
    M_ab = numerator*np.linalg.norm(ssb)/denom

    ya = ssa/np.linalg.norm(ssa)
    yb = ssb/np.linalg.norm(ssb)
    numerator = (
        sum([sum([p.K[i][j]*(ya[i]*yb[j] + yb[i]*ya[j])
                  * sum([yb[i]*ya[k]**2 - ya[i]*ya[k]*yb[k]
                         for k in range(len(ssa))])
                  for j in range(len(ssa))])
             for i in range(len(ssa))]) )
    denom = (
        sum([ya[i]**2 for i in range(len(ssa))])
        * sum([yb[i]**2 for i in range(len(ssa))])
        - sum([ya[i]*yb[i] for i in range(len(ssa))])**2 )
    # multiply by norm(ssa) because we are working in scaled variables
    M_ba = numerator*np.linalg.norm(ssa)/denom

    mu = np.array([mu_a, mu_b])
    M = np.array([[M_aa, M_ab], [M_ba, M_bb]])
    return mu, M

def get_separatrix_taylor_coeffs(s, order, dir_choice=1):
    """ Return a dictionary of Taylor coefficients for the unstable or stable
    manifolds of the semistable coexisting fixed point (x_a^*, x_b^*), up to
    order 'order'. Here I let (u, v) = (x_a^*, x_b^*) for notational
    convenience.  dir_choice = 0 returns the unstable manifold coefficients,
    dir_choice = 1 returns the stable manifold coefficients (i.e. dir_choice
    = 1 returns the separatrix). These coefficients are described in Eq 6 of
    the main text. """
    u, v = s.get_11_ss()
    coeffs = np.zeros(order)
    for i in range(order):
        if i == 0:
            coeffs[i] = v
            continue
        if i == 1:
            a = s.M[0][1]*u
            b = s.M[0][0]*u - s.M[1][1]*v
            c = -s.M[1][0]*v
            if dir_choice == 0:
                lin_val = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            else:
                lin_val = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            coeffs[i] = lin_val
            continue
        if i == 2:
            # Eq 37 of supplement. In my terms:
            # c_m/m! * alpha = c_m-1/(m-1)! * beta
            alpha = i*u*s.M[0][0] + (i+1)*u*s.M[0][1]*coeffs[1] - s.M[1][1]*v
            beta = ( s.M[1][0] + s.M[1][1]*coeffs[1] - (i-1)*s.M[0][0] -
                     (i-1)*s.M[0][1]*coeffs[1] )
            i_coeff = ( math.factorial(i) *
                        (coeffs[i-1]/math.factorial(i-1)*beta) ) / alpha
            coeffs[i] = i_coeff
            continue
        # Eq 38 of supplement. In my terms:
        # c_m/m! * alpha = c_m-1/(m-1)! * beta + sum_i=2^m-1 gamma[i]
        #alpha = i*u*p.M[0][0] + (i+1)*u*p.M[0][1]*coeffs[1]
        alpha = i*u*s.M[0][0] + (i+1)*u*s.M[0][1]*coeffs[1] - s.M[1][1]*v
        beta = ( s.M[1][0] + s.M[1][1]*coeffs[1] - (i-1)*s.M[0][0] -
                 (i-1)*s.M[0][1]*coeffs[1] )
        gamma = np.sum([ (coeffs[j]/(math.factorial(j) * math.factorial(i - j))
                          * (s.M[1][1]*coeffs[i-j]
                             - (i-j)*s.M[0][1]*coeffs[i-j]
                             - u*s.M[0][1]*coeffs[i-j+1]))
                        for j in range(2, i)])

        i_coeff = ( i / alpha * coeffs[i-1]*beta
                    + math.factorial(i) / alpha * gamma)
        coeffs[i] = i_coeff
    return coeffs

def get_lower_bound_of_separatrix(p, s, x_min, x_max, y_min, y_max,
                                  num_sections):
    """ Part of an iterative algorithm that numerically computes the
    separatrix, but uses adaptive sampling to only compute points nearby the
    separatrix (instead of far away from it). The range of points that you are
    investigating is [x_min, x_max, y_min, y_max]. This program assumes that
    at x_min the separatrix is larger than y_min, and at x_max it is smaller
    than y_max. This region will be cut along the x-axis into num_sections
    number of sections, and the y-axis will be cut so that the subregions are
    all squares. This function returns a list of points (that are ordered in
    the x-value) that delineate the lower bound of the separatrix, as well as
    the spatial step-size delta that was used to generate the subregions. """

    delta = (x_max - x_min)/num_sections # spatial variation
    N = num_sections + 1 # number of points needed to make num_sections
    xs = np.linspace(x_min, x_max, num=N)
    num_y_points = int(((y_max - y_min)/delta) + 1)
    ys = np.linspace(y_min, y_max, num=num_y_points)

    ssa = s.ssa
    ssb = s.ssb

    eps = 1e-3
    outcome_dict = {}
    for x in xs:
        for y in ys:
            ic = x*ssa + y*ssb
            t_out, y_out = solve(p, ic, 100)
            ss = y_out[-1]

            if np.linalg.norm(ss - ssa) < eps:
                outcome_dict[x, y] = 'a'
                #print('a', np.linalg.norm(ss - ssa))
            elif np.linalg.norm(ss - ssb) < eps:
                outcome_dict[x, y] = 'b'
                #print('b', np.linalg.norm(ss - ssb))
            elif x == 0 and y == 0:
                # edge case
                outcome_dict[x, y] = 'a'
            else:
                outcome_dict[x, y] = '?'
                #print('no steady state attained for {} {}. near a: {}. near b: {}'
                #      .format(x, y, np.linalg.norm(ss-ssa), np.linalg.norm(ss-ssb)))

    lower_bound_list = []
    for x in xs:
        flag = False
        for y in ys[::-1]:
            if flag is False:
                flag = outcome_dict[x, y]
            if flag == outcome_dict[x, y]:
                continue
            if flag != outcome_dict[x, y]:
                lower_bound_list.append([x, y])
                flag = outcome_dict[x, y]

    return lower_bound_list, delta

###############################################################################
##### FUNCTIONS THAT PRODUCE PLOTS
###############################################################################

def plot_original_and_SSR_trajectories(p, s, ax=None):
    """ Plots a SSR trajectory (characterized by SSR params 's') with initial
    condition ic_2d. Also plots the in-plane projection of the high-dimensional
    trajectory that starts at the corresponding initial condition in the
    high-dimensional space, as in Fig 2 of the main text. """
    if not ax:
        ax = plt.gca()
    #coord = [[.8, .2], [.9, .1]]
    coord = np.random.rand(10,2).tolist()
    for i,ic in enumerate(coord):
        ic_2d = ic
        ic_high = ic_2d[0]*s.ssa + ic_2d[1]*s.ssb

        t_2d, traj_2d = solve(s, ic_2d, 50)
        t_high, traj_high = solve(p, ic_high, 50)
        # project high-dimensional trajectory into the plane spanned by ssa and ssb
        traj_high_proj = p.project_to_2D(traj_high, s.ssa, s.ssb)
        """
        ax.plot(traj_high_proj[0,0], traj_high_proj[0,1], 'k.', zorder=3, ms=5)
        if i == 0:
            #ax.plot(traj_2d[:,0], traj_2d[:,1], color='blue', linewidth=0.5, label='2D trajectory')
            ax.plot(traj_high_proj[:,0], traj_high_proj[:,1], color='orange', linewidth=0.5,
                    label='high-dimensional trajectory')
        else:
            #ax.plot(traj_2d[:,0], traj_2d[:,1], color='blue', linewidth=0.5)
            ax.plot(traj_high_proj[:,0], traj_high_proj[:,1], color='orange', linewidth=0.5)
        """
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.savefig('SSR_demo_1.pdf')
    print('... saved figure to SSR_demo_1.pdf')

    return ax

def plot_2D_separatrix(p, s, ax=None):
    """ Plots the 2D separatrix, as given in Eq 6 of the main text """
    if not ax:
        ax = plt.gca()

    taylor_coeffs = get_separatrix_taylor_coeffs(s, order=100, dir_choice=1)
    # (u, v) is the coexistent fixed point (x_a^*, x_b^*) of the 2D system
    u, v = s.get_11_ss()
    xs = np.linspace(0, 1, 121)
    ys = [sum( [float(taylor_coeffs[j])*(xx-u)**j/math.factorial(j)
                for j in range(len(taylor_coeffs))] ) for xx in xs]

    ax.plot(xs, ys, color='k', lw=1, label='2D separatrix')
    plt.savefig('SSR_demo_2.pdf')
    print('... saved figure to SSR_demo_2.pdf')

    return ax


def plot_ND_separatrix(p, s, ax=None, sep_filename='11D_separatrix_1e-2.data',
                       color='b', y_max=2, label='separatrix', delta=0.01,
                       save_plot=False):
    """ Plots the in-plane 11D separatrix, which shows which steady state a
    point on the plane will tend towards. Sampling of points is done with a
    bisection-like method. """

    goal_delta = delta
    num_sections = 10
    if not ax:
        ax = plt.gca()

    prev_lower_bound_list, delta = (
        get_lower_bound_of_separatrix(p, s, x_min=0, x_max=1, y_min=0,
                                      y_max=y_max, num_sections=2) )
    prev_delta = delta

    # to save a calculated separatrix to the file, set load_data = True
    # to change the resolution of the calculated separatrix change delta
    load_data = True

    if load_data:
        with open(sep_filename, 'rb') as f:
            separatrix_lower_bound = pickle.load(f)
    else:
        print('RESOLUTION, NUMBER OF BISECTIONS')
        while delta > goal_delta:
            cumulative_lower_bound_list = []
            for i in range(len(prev_lower_bound_list)-1):
                x_min = prev_lower_bound_list[i][0]
                x_max = prev_lower_bound_list[i+1][0]
                if(x_min==x_max): break
                y_min = prev_lower_bound_list[i][1]
                y_max = prev_lower_bound_list[i+1][1] + prev_delta
                lower_bound_list, delta = (
                    get_lower_bound_of_separatrix(p, s, x_min=x_min, x_max=x_max,
                                        y_min=y_min, y_max=y_max, num_sections=2) )
                try:
                    if lower_bound_list[0] == cumulative_lower_bound_list[-1]:
                        cumulative_lower_bound_list.extend(lower_bound_list[1:])
                except IndexError:
                    cumulative_lower_bound_list.extend(lower_bound_list)
            prev_lower_bound_list = cumulative_lower_bound_list
            prev_delta = delta
            print(delta, len(cumulative_lower_bound_list))

            separatrix_lower_bound = np.array(cumulative_lower_bound_list)

        separatrix_lower_bound = np.array(cumulative_lower_bound_list)
        with open(sep_filename, 'wb') as f:
            pickle.dump(separatrix_lower_bound, f)
        print('(reduce computation time by setting load_data = True')

    ax.plot(separatrix_lower_bound[:,0], separatrix_lower_bound[:,1],
            color=color, label=label)
    if label:
        ax.legend()

    ax.legend(fontsize=12)
    if save_plot:
        plt.savefig('SSR_demo_3.pdf')
        print('... saved figure to SSR_demo_3.pdf')

    return ax
