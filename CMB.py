# Python script required to required to reproduce all numerical simulations presented in the paper "General heat features and thermodynamic control of active field theories"
# Source code by Artur Soriani
# In a terminal with python 3.10.2, use
# 'python CMB.py a0 atau b phibar betaF betaDelta'
# where
# 'a0' is the initial value of the external parameter;
# 'atau' is the final value of the external parameter;
# 'b' is the constant multiplying the phi^4 term in the free energy;
# 'phibar' is the conserved global density;
# 'betaF' is inverse temeprature times initial free energy of the homogeneous solution per lattice site;
# 'betaDelta' is inverse temperature times activity parameter.

import numpy as np
import time
import sys
import os
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline

def c(a,k):
    return a + 3*phibar**2 *b + kappa*k**2
    
def Omega(a,k): # relaxation frequency in the small noise expansion
     return k**z * (mob_phi* c(a,k) - gamma*Delta)

def inner(vec1,vec2): # standard inner product
    innerProd   = 0
    vec_size    = len(vec1)
    for alpha in range(0,vec_size):
        innerProd += vec1[alpha]*vec2[alpha]
    return innerProd #scalar
    
def gaussian(sigma): #generate zero-mean gaussian distribution
    return np.random.default_rng().normal(0,sigma,size=(dim+1,*gridSize))

# set the size of the momentum grid
def setkGrid():
    kGrid = np.zeros((dim,Nl))
    for alpha in range(0,dim): kGrid[alpha] = np.fft.fftfreq(Nl, d=dl/(2*np.pi))
    kGrid = np.array(np.meshgrid(*kGrid))
    return kGrid
    
def Fourier(sca): # Fourier transorm
    return np.fft.fftn(sca, norm='ortho')*np.sqrt(dl**dim)

def InverseFourier(sca_k): #Inverse Fourier transform
    return np.real(np.fft.ifftn(sca_k, norm='ortho'))/np.sqrt(dl**dim)

def grad_k(sca_k): #gradient operation in momentum space
    return 1j*kGrid* sca_k #vector

def div_k(vec_k): #divergence operation in momentum space
    return inner(1j*kGrid, vec_k) #scalar

def freeEnergy_k(phi,a): # coarse-grained free energy of phi4 models
    phi_k = Fourier(phi)
    return np.sum( a*phi**2/2 + b*phi**4/4 )*dl**dim + np.sum( kappa*inner(kGrid,kGrid)* np.real( np.conjugate(phi_k) * phi_k )/2 ) #scalar

def force_k(phi,a): # Fourier transform of the functional derivarive of the free energy
    phi_k       = Fourier(phi)
    phiCube_k   = Fourier(phi**3)
    return a*phi_k + b*phiCube_k + kappa*inner(kGrid,kGrid)* phi_k #scalar

def coupling_k(phi): # Fourier transform of the coupling between active system and fuel reservoir
    phi_k = Fourier(phi)
    return gamma* grad_k(phi_k) #vector

def spuriousDrifts_k(phi): # Fourier transform of the spurious drifts
    spu_phi_k   = np.zeros((dim,*gridSize))
    spu_n_k     = np.zeros(gridSize)
    
    return spu_phi_k, spu_n_k #vector, scalar
    
def relax_k(phi, a): # time-independent realization of a time step
    # generate uncorrelated noise in momentum space
    gau_phi     = gaussian(1)[:dim]
    gau_phi_k   = Fourier(gau_phi)
    noi_phi_k   = np.sqrt(2*mob_phi/(dt*dl**dim))* gau_phi_k

    # calculate quantities in the dynamic equation
    frc_k               = force_k(phi,a)
    cpl_k               = coupling_k(phi)
    spu_phi_k, spu_n_k  = spuriousDrifts_k(phi)

    # calculate current
    crt_k = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
    
    # update field
    phiNew  = phi - InverseFourier(div_k(crt_k))*dt
    
    # calculate field at half time step
    phiHalf = phi - InverseFourier(div_k(crt_k))*dt/2

    # calculate quantities at half time step
    frc_k               = force_k(phiHalf,a)
    cpl_k               = coupling_k(phiHalf)
    spu_phi_k, spu_n_k  = spuriousDrifts_k(phiHalf)
    
    # calculate current at half time step
    crt_k = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
    
    # calculate observables for later integration
    P = Delta/mob_phi * np.sum( np.real(inner( np.conjugate(cpl_k), crt_k - Delta*cpl_k )) )
    
    return phiNew, P #scalar
    
def manip_k(phi, t): # time-dependent realization of a time step
    # generate uncorrelated noise in momentum space
    gau_phi     = gaussian(1)[:dim]
    gau_phi_k   = Fourier(gau_phi)
    noi_phi_k   = np.sqrt(2*mob_phi/(dt*dl**dim))* gau_phi_k

    # calculate quantities in the dynamic equation
    frc_k               = force_k(phi,aProtocol(t))
    cpl_k               = coupling_k(phi)
    spu_phi_k, spu_n_k  = spuriousDrifts_k(phi)
    
    # calculate current
    crt_k = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
    
    # update field
    phiNew  = phi - InverseFourier(div_k(crt_k))*dt
    
    # calculate field at half time step
    phiHalf = phi - InverseFourier(div_k(crt_k))*dt/2

    # calculate quantities at half time step
    frc_k               = force_k(phiHalf,aProtocol(t+dt/2))
    cpl_k               = coupling_k(phiHalf)
    spu_phi_k, spu_n_k  = spuriousDrifts_k(phiHalf)
    
    # calculate current at half time step
    crt_k = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
    
    # calculate observables for later integration
    dF  = np.sum( phiHalf**2/2 )*dl**dim
    P   = Delta/mob_phi * np.sum( np.real(inner( np.conjugate(cpl_k), crt_k - Delta*cpl_k )) )
    
    return phiNew, P, dF #scalar
    
RKcoef1 = [0/6, 3/6, 3/6, 6/6] # Runge-Kutta coefficients
RKcoef2 = [1/6, 2/6, 2/6, 1/6] # Runge-Kutta coefficients
    
def relaxRK4_k(phi, a):
    gau_phi   = gaussian(1)[:dim]
    gau_phi_k = Fourier(gau_phi)
    noi_phi_k = np.sqrt(2*mob_phi/(dt*dl**dim))* gau_phi_k

    phiNew  = phi
    phiHalf = phi
    phidot  = np.zeros((5,*gridSize))

    for i in range(0,4):
        frc_k                = force_k(          phi + phidot[i]*RKcoef1[i]*dt , a)
        cpl_k                = coupling_k(       phi + phidot[i]*RKcoef1[i]*dt )
        spu_phi_k, spu_n_k   = spuriousDrifts_k( phi + phidot[i]*RKcoef1[i]*dt )
        
        crt_k       = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
        phidot[i+1] = - InverseFourier(div_k(crt_k))
        
        phiNew  += phidot[i+1]*RKcoef2[i]*dt
        phiHalf += phidot[i+1]*RKcoef2[i]*dt/2
        
    frc_k              = force_k(phiHalf,a)
    cpl_k              = coupling_k(phiHalf)
    spu_phi_k, spu_n_k = spuriousDrifts_k(phiHalf)
    
    crt_k = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
    
    P  = Delta/mob_phi * np.sum( np.real(inner( np.conjugate(cpl_k), crt_k - Delta*cpl_k )) )

    return phiNew, P # scalar
    
def manipRK4_k(phi, t):
    gau_phi   = gaussian(1)[:dim]
    gau_phi_k = Fourier(gau_phi)
    noi_phi_k = np.sqrt(2*mob_phi/(dt*dl**dim))* gau_phi_k

    phiNew  = phi
    phiHalf = phi
    phidot  = np.zeros((5,*gridSize))

    for i in range(0,4):
        frc_k                = force_k(          phi + phidot[i]*RKcoef1[i]*dt , aProtocol(t+RKcoef1[i]*dt))
        cpl_k                = coupling_k(       phi + phidot[i]*RKcoef1[i]*dt )
        spu_phi_k, spu_n_k   = spuriousDrifts_k( phi + phidot[i]*RKcoef1[i]*dt )
        
        crt_k       = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
        phidot[i+1] = - InverseFourier(div_k(crt_k))
        
        phiNew  += phidot[i+1]*RKcoef2[i]*dt
        phiHalf += phidot[i+1]*RKcoef2[i]*dt/2
    
    frc_k              = force_k(phiHalf,aProtocol(t+dt/2))
    cpl_k              = coupling_k(phiHalf)
    spu_phi_k, spu_n_k = spuriousDrifts_k(phiHalf)
    
    crt_k = - mob_phi*grad_k(frc_k) + Delta*cpl_k + noi_phi_k/beta**(1/2) + spu_phi_k/beta
    
    dF = np.sum( phiHalf**2/2 )*dl**dim
    P  = Delta/mob_phi * np.sum( np.real(inner( np.conjugate(cpl_k), crt_k - Delta*cpl_k )) )

    return phiNew, P, dF #scalar
    
# steady-state averages and integrated relaxation functions, needed for calculating the optimal protocols
def phi2ss(a,k):     return + mob_phi*k**z/(Omega(a,k))
def pss(a,k):        return - (gamma*Delta)**2 *k**z /(beta*mob_phi) * phi2ss(a,k)
def zeta11dF(a,k):   return + 1/(2*beta) * phi2ss(a,k) * (mob_phi*k**z/(2*Omega(a,k)**2))
def zeta22P(a,k):    return - gamma*Delta*k**z *c(a,k)/beta * phi2ss(a,k) * ((mob_phi*k**z)**2/(2*Omega(a,k)**4))
def zeta21Pdif(a,k): return + gamma*Delta*k**z/beta * (mob_phi*k**z)**2/(4*Omega(a,k)**4) * ( 1 - mob_phi*k**z *c(a,k)/Omega(a,k) )

def Pss(a):
    result = 0
    for k in kList: result += 2*pss(a,k)
    return result

def mDRT(a,k):
    return zeta11dF(a,k) + zeta22P(a,k) - zeta21Pdif(a,k)

def MDRT(a):
    result = 0
    for k in kList: result += 2*mDRT(a,k)
    return result

def tauOP(INT, ai, af):
    integral = quad(lambda a: np.sqrt( MDRT(a)/(INT + Pss(a)) ), ai, af )
    return integral[0]

# generates the lists used to interpolate the optimal protocols
def generateOptimalList(tau):
    opPoints = max(100,20*tau/taumax)
    optimalListS, optimalListA, optimalListAdot = [], [], []
    aop = a0
    if(tau<taumax):
        
        imin = -Pss(amin)
        imax = imin + ( quad(lambda a: np.sqrt(MDRT(a)), amin, amax )[0]/tau )**2
        INT = brentq( lambda i: tauOP(i,amin,amax) - tau, imin, imax)
    
        if atau>a0:
            while(aop<atau):
                optimalListS.append( +tauOP(INT,a0,aop)/tau )
                optimalListA.append( aop )
                optimalListAdot.append( +np.sqrt( (INT + Pss(aop))/MDRT(aop) )*tau )
                aop += (atau-a0)/(opPoints-1)
            optimalListS.append( +tauOP(INT,a0,atau)/tau )
            optimalListA.append( atau )
            optimalListAdot.append( +np.sqrt( (INT + Pss(atau))/MDRT(atau) )*tau )
        else:
            while(aop>atau):
                optimalListS.append( -tauOP(INT,a0,aop)/tau )
                optimalListA.append( aop )
                optimalListAdot.append( -np.sqrt( (INT + Pss(aop))/MDRT(aop) )*tau )
                aop += (atau-a0)/(opPoints-1)
            optimalListS.append( -tauOP(INT,a0,atau)/tau )
            optimalListA.append( atau )
            optimalListAdot.append( -np.sqrt( (INT + Pss(atau))/MDRT(atau) )*tau )
            
    else:
        
        ainf = gamma*Delta/mob_phi - 3*b*phibar**2 - kappa*kmin**2
        alower = ( 1-np.exp(0.20*(1-tau/taumax)) ) *ainf
        ahighr = (   np.exp(0.01*(1-tau/taumax)) ) *amin
        a1 = brentq( lambda a: - tauOP(-Pss(a),a0,a) + tauOP(-Pss(a),a,atau) - tau, alower, ahighr )
        INT = -Pss(a1)
        
        while(aop>a1):
            optimalListS.append( -tauOP(INT,a0,aop)/tau )
            optimalListA.append( aop )
            optimalListAdot.append( -np.sqrt( (INT + Pss(aop))/MDRT(aop) )*tau )
            aop -= 2*(amax-amin)/(opPoints-1)
        tauOPa1 = -tauOP(INT,a0,a1)
        optimalListS.append( tauOPa1/tau )
        optimalListA.append( a1 )
        optimalListAdot.append( 0 )
        aop = a1 + (amax-amin)/(opPoints-1)
        while(aop<atau):
            optimalListS.append( tauOPa1/tau + tauOP(INT,a1,aop)/tau )
            optimalListA.append( aop )
            optimalListAdot.append( +np.sqrt( (INT + Pss(aop))/MDRT(aop) )*tau )
            aop += 2*(amax-amin)/(opPoints-1)
        optimalListS.append( tauOPa1/tau + tauOP(INT,a1,atau)/tau )
        optimalListA.append( atau )
        optimalListAdot.append( +np.sqrt( (INT + Pss(atau))/MDRT(atau) )*tau )
    
    return optimalListS, optimalListA, optimalListAdot

######################
# run the simulation #
######################

# define the protocol
def aProtocol(t): # return interpA(t/tau)
    a1st = gamma*Delta/mob_phi - b*phibar**2
    if interpA(t/tau) > a1st : return interpA(t/tau)
    else                     : return a1st
def adotProtocol(t): # return interpAdot(t/tau)/tau
    a1st = gamma*Delta/mob_phi - b*phibar**2
    if interpA(t/tau) > a1st : return interpAdot(t/tau)/tau
    else                     : return 0

# define system constants
z       = 2                     # dynamic critical exponent, 0 for model A and 2 for model B
dim     = 1                     # dimensions
dl      = 2**(-0)               # lattice constant
Nl      = int(2**(+6))          # lattice sites
mob_phi = 10**(+0)              # system mobility
phibar  = float(sys.argv[4])    # global density

a0, atau            = float(sys.argv[1]), float(sys.argv[2])    # initial and final values of external parameter
b, kappa            = float(sys.argv[3]), 1                     # free energy constants
betaF, betaDelta    = float(sys.argv[5]), float(sys.argv[6])    # unitless quantities used to determine temperature and activity

mob_n   = mob_phi                                           # fuel mobility
gamma   = mob_phi                                           # coupling constant
vol     = (Nl*dl)**dim                                      # volume
beta    = betaF/((a0*phibar**2/2 + b*phibar**4/4) *dl**dim) # inverse temperature
Delta   = betaDelta/beta                                    # activity parameter

# define simulation constants
Np          = 1         # number of paths
tauPoints   = 13        # number of points in the tau plot
dt          = 10**(-3)  # time step size
xmin        = 10**(-2)  # approximate minimum horizontal value in the tau plot
xmax        = 10**(+2)  # approximate maximum horizontal value in the tau plot

# set the size of the space grid
gridSize = []
for alpha in range(0,dim): gridSize.append(Nl) #alpha = 0,1,...,dim-1 represents cartesian coordinates

# set the size of the momentum grid
kGrid = setkGrid()

# calculate useful quantities
kList       = 2*np.pi/(Nl*dl) * np.arange(1, Nl/2+1)    # allowed values of momentum
kmin, kmax  = kList[0], kList[-1]                       # minimum and maximum values of momentum
amin, amax  = min([a0,atau]), max([a0,atau])            # minimum and maximum values of the external parameter
Omega_min   = Omega(amin,kmin)                          # minimum relaxation frequency
taumax      = tauOP(-Pss(amin),amin,amax)               # maximum value of tau for monotonic protocols

# calculate number of iterations for beginning and ending relaxations
Nr  = int(np.floor((10/Omega(a0  ,kmin))/dt))
NR  = int(np.floor((10/Omega(atau,kmin))/dt))

# save time variable to calculate program time at the end
start = time.time()

# define where to save the data
outputPath = 'a0='+str('{0:g}'.format(a0))+',atau='+str('{0:g}'.format(atau))+',b='+str('{0:g}'.format(b))+',phibar='+str('{0:g}'.format(phibar))+',betaF='+str('{0:g}'.format(betaF))+',betaDelta='+str('{0:g}'.format(betaDelta))
if not os.path.exists(outputPath): os.makedirs(outputPath)
if not os.path.exists(outputPath+'/optimalProtocols'): os.makedirs(outputPath+'/optimalProtocols')


# save parameters
paramFile = open(outputPath+'/parameters.dat','a')
if os.path.getsize(outputPath+'/parameters.dat') == 0:
    paramFile.write('z='+str(z)+'\ndim='+str(dim)+'\ndl='+str(dl)+'\nNl='+str(Nl)+'\nmob_phi='+str(mob_phi)+'\nphibar='+str(phibar)+'\n')
    paramFile.write('a0='+str(a0)+'\natau='+str(atau)+'\nb='+str(b)+'\nkappa='+str(kappa)+'\nbetaF='+str(betaF)+'\nbetaDelta='+str(betaDelta)+'\n')
    paramFile.write('mob_n='+str(mob_n)+'\ngamma='+str(gamma)+'\nvol='+str(vol)+'\nbeta='+str(beta)+'\nDelta='+str(Delta)+'\n')
    paramFile.write('Np='+str(Np)+'\ndt='+str(dt)+'\ntauPoints='+str(tauPoints)+'\nxmin='+str(xmin)+'\nxmax='+str(xmax)+'\n')
    paramFile.flush()
paramFile.close()
    
x     = xmin

# initialize phi
phi0 = np.ones((Np,*gridSize))*phibar
    
# loop until phi reaches steady state
for n in range(0,Nr):
    for p in range(0,Np): phi0[p] = relax_k(phi0[p], a0)[0]

#calculate initial free energy
intEzero = np.zeros(Np)
for p in range(0,Np): intEzero[p] = freeEnergy_k(phi0[p],a0)

# start the main loop, one iteration for each value of tau
for i in range(0,tauPoints):
    
    # calculate process duration and amount of timesteps
    Nt  = int(np.floor((x/Omega_min)/dt))
    tau = Nt*dt
    
    # generate optimal protocols or load them if saved
    if not os.path.isfile(outputPath+'/optimalProtocols/x='+str(round(Omega_min*tau,6))+'.npz'):
        optimalListS, optimalListA, optimalListAdot = generateOptimalList(tau)
        np.savez(outputPath+'/optimalProtocols/x='+str(round(Omega_min*tau,6))+'.npz', optimalListS=optimalListS, optimalListA=optimalListA, optimalListAdot=optimalListAdot)
    else:
        optimalList     = np.load(outputPath+'/optimalProtocols/x='+str(round(Omega_min*tau,6))+'.npz')
        optimalListS    = optimalList['optimalListS']
        optimalListA    = optimalList['optimalListA']
        optimalListAdot = optimalList['optimalListAdot']
    
    # interpolate the optimal protocols with cubic polynomials
    interpA    = CubicSpline(optimalListS, optimalListA)
    interpAdot = CubicSpline(optimalListS, optimalListAdot)
    
    # load initial time and phi
    t   = 0
    phi = np.zeros((Np,*gridSize))
    for p in range(0,Np): phi[p] = phi0[p]

    # initialize necessary quantities to calculate the energy flows
    dF      = np.zeros(Np)
    P       = np.zeros(Np)
    extW    = np.zeros(Np)
    actW    = np.zeros(Np)
      
    # given a value of tau, loop from t = 0 to t = tau
    for n in range(0,Nt):
        for p in range(0,Np):
            # solving the dynamics
            phi[p], P[p], dF[p] = manip_k(phi[p], t)
            
            # calculate works
            extW[p] += ( adotProtocol(t+dt/2) * dF[p] ) * dt    # Strato time integral (mid-point rule)
            actW[p] += ( mob_n*vol*Delta**2   +  P[p] ) * dt    # Strato time integral (mid-point rule)
        
        # update time for next iteration
        t += dt
        
    # calculate phibar, free energy at tau and process heat
    phiAvg  = np.zeros(Np)
    intEtau = np.zeros(Np)
    prcQ    = np.zeros(Np)
    for p in range(0,Np):
        phiAvg[p]   = np.sum( phi[p] )/vol
        intEtau[p]  = freeEnergy_k(phi[p],atau)
        prcQ[p]     = extW[p] - (intEtau[p] - intEzero[p]) + actW[p]
    
    # initialize more energy flows
    intEtauR    = np.zeros(Np)
    pstW        = np.zeros(Np)
    totQ        = np.zeros(Np)
    
    # # loop until phi reaches steady state
    for n in range(0,NR):
        for p in range (0,Np):
            phi[p], P[p] = relax_k(phi[p], atau)
            pstW[p] += ( mob_n*vol*Delta**2 + P[p] ) * dt # Strato time integral
    
    # calculate free energy at tau+tauR and total heat
    for p in range(0,Np):
        intEtauR[p] = freeEnergy_k(phi[p],atau)
        totQ[p]     = extW[p] - (intEtauR[p] - intEzero[p]) + actW[p] + pstW[p]
    
    # prepare file for saving data
    xFile = open(outputPath+'/x='+str(round(Omega_min*tau,6))+'.dat','a')
    if os.path.getsize(outputPath+'/x='+str(round(Omega_min*tau,6))+'.dat') == 0:
        xFile.write('1.phiAvg     2.intEzero   3.intEtau    4.intEnss    5.extW       6.actW       7.prcQ       8.intEtauR   9.intEss     10.pstW      11.totQ\n')
        xFile.flush()
    
    # save data
    for p in range(0,Np):
        xFile.write( str( '{:e}'.format(          phiAvg[p] ) )              + ' '  ) #1.phiAvg
        xFile.write( str( '{:e}'.format(  beta* intEzero[p] ) )              + ' '  ) #2.intEzero
        xFile.write( str( '{:e}'.format(  beta*  intEtau[p] ) )              + ' '  ) #3.intEtau
        xFile.write( str( '{:e}'.format(  beta* (intEtau[p]-intEzero[p]) ) ) + ' '  ) #4.intEnss
        xFile.write( str( '{:e}'.format(  beta*     extW[p] ) )              + ' '  ) #5.extW
        xFile.write( str( '{:e}'.format(  beta*     actW[p] ) )              + ' '  ) #6.actW
        xFile.write( str( '{:e}'.format(  beta*     prcQ[p] ) )              + ' '  ) #7.prcQ
        xFile.write( str( '{:e}'.format(  beta* intEtauR[p] ) )              + ' '  ) #8.intEtauR
        xFile.write( str( '{:e}'.format(  beta*(intEtauR[p]-intEzero[p]) ) ) + ' '  ) #9.intEss
        xFile.write( str( '{:e}'.format(  beta*     pstW[p] ) )              + ' '  ) #10.pstW
        xFile.write( str( '{:e}'.format(  beta*     totQ[p] ) )              + '\n' ) #11.totQ
        xFile.flush()
    xFile.close()
    
    # update tau for next iteration
    x *= (xmax/xmin)**(1/(tauPoints-1))

# calculate program time and print    
end = time.time()
if   ( (end - start)    < 60)   : time_out = str( round((end - start)      ,2) ) + ' seconds.'
elif ( (end - start)/60 < 60)   : time_out = str( round((end - start)/60   ,2) ) + ' minutes.'
else                            : time_out = str( round((end - start)/60**2,2) ) + ' hours.'
print('Simulation time:',time_out,flush=True)