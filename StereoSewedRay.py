import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1' 
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from time import time
from sys import stdout

begin = time()

seed=np.random.randint(1000000)
print(seed)
rng=np.random.default_rng(seed)

s=2
n=2*s

################################################################################
dt=0.5
lam=4.0
omega=1.0

def V0dtsq(phi):
    return dt**2*(0.5*phi**2+lam*phi**4/24)
def V1dtsq(phi):
    return dt**2*(phi+lam*phi**3/6)
def V2dtsq(phi):
    return dt**2*(1+lam*phi**2/2)
def W0dtsq(K):
    return dt**2*K[0]**3/6*(lam*CL[1])
def W1dtsq(K):
    return dt**2*K[0]**2/2*(lam*CL[1])
def W2dtsq(K):
    return dt**2*K[0]*(lam*CL[1])

################################################################################
def progressBar(count_value, total, suffix=''):
    bar_length = 20
    filled_up_Length = int(bar_length* count_value / total)
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * (filled_up_Length-1)+'>' + '-' * (bar_length - filled_up_Length)
    stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    stdout.flush()

class odeBreak:
    C=np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
    A=np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]])
    B=np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E=np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40])
    
    SAFETY=0.9
    rtol=1.0e-07
    atol=1.0e-08

    def __init__(self, fun, t_ini, break_fun):
        self.fun=fun
        self.t_ini=t_ini
        self.bfun=break_fun

    def RKF45(self, t_bound, y_ini):
        t=self.t_ini
        Y=y_ini
        h=0.01
        h_max=0.1
        K=np.zeros((7,len(y_ini)))+0j
        
        toRun=True
        if (t-t_bound) >= 0:
            toRun=False
    
        k0_old=self.fun(t,Y)

        while toRun:
            t_new = t + h
            if (t_new - t_bound) > 0:
                t_new = t_bound
            h = t_new - t
        
            K[0] = k0_old
            for s, (a, c) in enumerate(zip(self.A[1:], self.C[1:]), start=1):
                dy=np.dot(K[:s].T,a[:s])*h
                K[s]=self.fun(t+c*h,Y+dy)
        
            y_new=Y+h*np.dot(K[:-1].T,self.B)
            K[-1]=self.fun(t+h,y_new)
        
            scale=self.atol+np.maximum(np.abs(Y),np.abs(y_new))*self.rtol
            error_norm=np.linalg.norm(np.dot(K.T,self.E)*h/scale)/(self.E).size**0.5
        
            if error_norm < 1:
                step_accepted = True
                t = 0+t_new
                Y = 0+y_new
                k0_old=0+K[-1]
                if self.bfun(t,Y)>0:
                    break 
            else:
                step_accepted = False
        
            factor = self.SAFETY * (error_norm+1.0e-15) ** (-0.2)
            if factor <= 0.5:
                factor = 0.5  
            elif factor >= 2.0:
                factor =   2.0 
            h *= factor

            if h>h_max:
                h = 0+h_max
        
            if step_accepted and (t-t_bound)>=0:
                toRun=False
    
        return [t,Y]
################################################################################

def funI(K):
    return -1j/(2*dt)*(4*K[0]*CL[2]+(-2*K[1]+2*K[n-1])*CL[1]-4*W0dtsq(K)
        +(K[1]-K[0])**2-(K[n-1]+K[0])**2
        +np.sum([(K[j+1]-K[j])**2-(K[n-j]-K[n-j-1])**2-2*(V0dtsq(K[j])-V0dtsq(K[n-j])) for j in range(1,s)]))

def ConjPartialI(Z):
    tmp=np.zeros(n)*1j
    tmp[0]=Z[1]+Z[n-1]-2*CL[2]+2*W1dtsq(Z)
    tmp[1]  = Z[2]  -2*Z[1]  +CL[1]+V1dtsq(Z[1])  +Z[0]
    tmp[n-1]=-Z[n-2]+2*Z[n-1]-CL[1]-V1dtsq(Z[n-1])+Z[0]
    for j in range(2,s):
        tmp[j]=Z[j+1]-2*Z[j]+Z[j-1]+V1dtsq(Z[j])
        i=n-j
        tmp[i]=-(Z[i+1]-2*Z[i]+Z[i-1]+V1dtsq(Z[i]))
    tmp[s]=Z[s-1]-Z[n-s+1]
    return np.conj(1j/dt*tmp)

def ConjHessianV(Z,V):
    f=np.zeros(n)*1j
    f[0]=2*W2dtsq(Z)
    for i in range(1,s):
        f[i]  = -2+V2dtsq(Z[i])
        f[n-i]=  2-V2dtsq(Z[n-i])

    tmp=np.zeros(n)*1j
    for i in range(1,s):
        tmp[i]=V[i-1]+f[i]*V[i]+V[i+1]
    for i in range(s+1,n-1):
        tmp[i]=-V[i-1]+f[i]*V[i]-V[i+1]
    tmp[0]=f[0]*V[0]+V[1]+V[n-1]
    tmp[n-1]=V[0]-V[n-2]+f[n-1]*V[n-1]
    tmp[s]=V[s-1]-V[s+1]
    return np.conj(1j/dt*tmp)

def O(K):
    return K[0]*(K[1:s+1]+np.flip(K[s:n]))

def Initial(r):
    B=np.zeros((n,n))
    
    c0=np.dot(iE,iE)+1
    c2=(1/c0)**2

    B0 = -4*np.outer(iE,iE)
    for i in range(n-1):
        B0[i,i] += 2*c0
    B0 *= c2

    B[:(n-1),1:]=0+B0
    B[:,0]=vecB
    B[-1,1:] = 4*c2*iE

    crk = p0*r**(kappa)
    PhV=np.zeros((n,n))*1j
    for j in range(n):
        PhV[j]=np.sum([crk[i]*v[i]*B[i,j] for i in range(n)],axis=0)
    PhV[0] += Phi0

    return PhV.flatten()

def IJ(K):
    I=funI(K)
    J=np.array([ConjPartialI(K[:n]),*([K[i*n:(i+1)*n] for i in range(1,n)])]).T
    sign,lndet=np.linalg.slogdet(J)
    return np.array([np.real(I),np.imag(I),lndet,np.angle(sign)])

def OuterFun(t,K):
    Ire,Iim,lndetJ,phaseJ=IJ(K)
    outer=np.exp(-Ire+lndetJ-1j*(Iim-phaseJ))
    return np.concatenate((ConjPartialI(K[:n]),*([ConjHessianV(K[:n],K[i*n:(i+1)*n]) for i in range(1,n)]),[np.abs(outer),outer],outer*O(K)),axis=0)

def OuterBreakFun(t,K):
    Ire,Iim,lndetJ,phaseJ=IJ(K)
    expo=(Ire-lndetJ)
    if (expo-expo0>40.0*n) or (expo<-700.0):
        if (expo<-700.0):
            print("Possible divergence!!!",iE)
        return 1
    else:
        return -1

def InnerFun(r, K):
    crk = p0*r**(kappa)
    Phi=Phi0+np.sum([crk[i]*v[i]*vecB[i] for i in range(n)],axis=0)

    inner = det1*r**(sumkappam1)*np.exp(-funI(Phi))
    return np.concatenate(([np.abs(inner),inner],inner*O(Phi)),axis=0)

def InnerBreakFun(r, K):
    return -1

def RayIntegral():
    res0=ode1.RKF45(r0,y_ini=np.zeros(LastThree))
    ZVP=np.concatenate((Initial(r0),np.zeros(LastThree)),axis=0)
    Ire,Iim,lndetJ,phaseJ=IJ(ZVP)
    global expo0
    expo0=(Ire-lndetJ)
    res1=ode2.RKF45(big_number,y_ini=ZVP)
    tmp=res0[1][-LastThree:]+res1[1][-LastThree:]
    if np.abs(tmp[0]) != 0.0:
        tmp[1:] = tmp[1:]/tmp[0]
    return np.array(tmp)

def UpdateE():
    global vecB
    global det1
    
    c0=np.dot(iE,iE)+1
    c1=1/c0

    vecB[:(n-1)] = 2*c1*iE
    vecB[-1] = c1*(c0-2)
    det1 = det0*2**(n-1)*c1**(n+1)*(kappa[-1]*(c0-2)**2+4*np.sum(kappa[:(n-1)]*iE**2))

################################################################################
p0=1.0
r0=1.0

phi0=rng.normal(0.0,np.sqrt(0.5/omega))
pi0=rng.normal(0.0,np.sqrt(0.5*omega))

CL=np.zeros(s+2)
#CL[0]=phi0
#CL[1]=pi0*dt+phi0*(1-0.5*(omega*dt)**2)
CL[0]=-0.4321466 
CL[1]=-0.1166508
for i in range(1,s+1):
    CL[i+1]=2*CL[i]-CL[i-1]-V1dtsq(CL[i])

iname="_classical"
np.savetxt(iname,CL)
print(f"The classical configuration is saved to: {iname}")

h0=np.zeros((n,n))
for i in range(0,s):
    h0[i,i+1]=1
    h0[i+1,i]=1
for i in range(s,n-1):
    h0[i,i+1]=-1
    h0[i+1,i]=-1
h0[0,-1]=1
h0[-1,0]=1

Phi0=np.zeros(n)
for i in range(1,s):
    resident=-2+V2dtsq(CL[i+1])
    h0[i,i]=resident
    Phi0[i]=CL[i+1]
    j=n-i
    h0[j,j]=-resident
    Phi0[j]=CL[i+1]
Phi0[s]=CL[s+1]
NMeasure=len(O(Phi0))
LastThree=2+NMeasure

w0,v0=np.linalg.eigh(h0)
v0=np.transpose(v0)
kappa=np.zeros(n)
v=np.zeros((n,n))*1j
kappa[:s]=-w0[:s]/dt
v[:s]=np.sqrt(0.5)*(1+1j)*v0[:s]
kappa[s:]=w0[s:]/dt
v[s:]=np.sqrt(0.5)*(1-1j)*v0[s:]

sind=np.argsort(kappa)
kappa=kappa[sind]
v=v[sind]
det0=np.linalg.det(v)*p0**n
det1=1.0
sumkappam1=np.sum(kappa)-1

period=np.ones(n-1)*np.pi
period[-1] += np.pi

vecB=np.zeros(n)
################################################################################

big_number=20
expo0=0.0

ode1=odeBreak(fun=InnerFun,t_ini=0.0,break_fun=InnerBreakFun)
ode2=odeBreak(fun=OuterFun,t_ini=np.log(r0),break_fun=OuterBreakFun)

Natt=1000000
CheckPerRound=200
NRound=5
BurnIn=NRound*CheckPerRound
saveEvery=10
optimal_rate=0.3


scale=0.3
pr=0
fu=1
Weight=np.zeros(2)*0j+10**(-40)

E = rng.normal(0,scale,n-1)
iE=E+0
UpdateE()
Outcome=RayIntegral()
acc = 0+Outcome
Weight[pr]=Outcome[0]
print("Initial weight is %.3e"%(np.real(Weight[pr])))

save=[]
iacc=0
isave=1
bcount=0

print("This starts the burn-in period, where the step size will be adjusted automatically.")
for i in range(1,Natt+BurnIn+1):
    delta = rng.normal(0,scale,n-1)
    iE=E+delta
    UpdateE()
    Outcome=RayIntegral()
    Weight[fu] =Outcome[0]
    #if np.real(Weight[fu])>100000:
    #    print(np.real(Weight[fu]),iE)

    if rng.uniform(0.0,1.0) < np.real(Weight[fu]/Weight[pr]):
        E = 0+iE
        acc = 0+Outcome
        pr, fu = fu, pr
        iacc += 1

    if i <= BurnIn:
        #progressBar(i,BurnIn)
        if i%CheckPerRound == 0:
            irate=iacc/CheckPerRound
            scale *= np.exp(2*CheckPerRound/i*(irate-optimal_rate))

            bcount += 1    
            if bcount == NRound:
                print(f"The final scale is {scale}.")
                print("\n")
                print("The data will be saved every %d step(s)."%saveEvery)
            iacc=0
    else:
        if isave % saveEvery == 0:
            save.append(0+acc) 
        #progressBar(isave,Natt)
        isave += 1

save=np.array(save)
print("\n")
print("The acceptence rate: %.3f"%(iacc/Natt))

iname="_MarkovChain"
np.savetxt(iname,save)
print(f"The Markov Chain is saved to: {iname}")

end = time()
print("The MC takes %f seconds."%(end-begin))
