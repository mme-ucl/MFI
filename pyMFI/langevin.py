import numpy as np

def run_2D(pace=100, nsteps=100000, sigma=0.1, height=0.5, biasfactor=10, ipos=np.array([-1,-1]),tag=1):
    with open("plumed.dat","w") as f:
        print("""p: DISTANCE ATOMS=1,2 COMPONENTS
ff: MATHEVAL ARG=p.x,p.y PERIODIC=NO FUNC=(7*x^4-23*x^2+7*y^4-23*y^2)
bb: BIASVALUE ARG=ff
METAD ARG=p.x,p.y PACE={} SIGMA={},{} HEIGHT={} GRID_MIN=-3,-3 GRID_MAX=3,3 GRID_BIN=300,300 BIASFACTOR={} TEMP=120 FILE=HILLS_{}
PRINT FILE=position_{} ARG=p.x,p.y STRIDE=10""".format(pace, sigma, sigma, height, biasfactor,tag,tag),file=f)

    with open("input","w") as f:
        print("""temperature 1
tstep 0.005
friction 1
dimension 2
nstep {}
ipos {},{}
periodic false""".format(nsteps,ipos[0],ipos[1]),file=f)
    

def run_2D_Invernizzi(pace=200, nsteps=100000, sigma=0.1, height=0.5, biasfactor=10, ipos=np.array([-1,-1]),tag=1):
    with open("plumed.dat","w") as f:
        print("""p: DISTANCE ATOMS=1,2 COMPONENTS
ff: MATHEVAL ARG=p.x,p.y PERIODIC=NO FUNC=(1.34549*x^4+1.90211*x^3*y+3.92705*x^2*y^2-6.44246*x^2-1.90211*x*y^3+5.58721*x*y+1.33481*x+1.34549*y^4-5.55754*y^2+0.904586*y+18.5598)
bb: BIASVALUE ARG=ff
METAD ARG=p.x,p.y PACE={} SIGMA={},{} HEIGHT={} GRID_MIN=-3,-3 GRID_MAX=3,3 GRID_BIN=300,300 BIASFACTOR={} TEMP=120 FILE=HILLSinve_{}
PRINT FILE=positioninve_{} ARG=p.x,p.y STRIDE=10""".format(pace, sigma, sigma, height, biasfactor,tag,tag),file=f)

    with open("input","w") as f:
        print("""temperature 1
tstep 0.005
friction 10
dimension 2
nstep {}
ipos {},{}
periodic false""".format(nsteps,ipos[0],ipos[1]),file=f)
