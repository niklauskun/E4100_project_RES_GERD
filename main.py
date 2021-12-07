import cvxpy as cp
import numpy as np
import pandas as pd

## set up scenario
Lmbd1 = 0.4 # penalty of flow requirement violation ($/m^3)
case = 1 # one week 0, 5 years 1

## Load data in hourly resolution
inflow = pd.read_csv('inflow.csv', header = None) # unit m^3/s
inflow_full = pd.read_csv('inflow_full.csv', header = None) # unit m^3/s
load = pd.read_csv('load.csv', header = None) # normalized, sum up to 8760 for each year
solar = pd.read_csv('solar.csv', header = None)
wind = pd.read_csv('wind.csv', header = None)
prec = pd.read_csv('prec.csv', header = None) # unit kg/m^2/s
evap = pd.read_csv('evap.csv', header = None) # unit kg/m^2/s

## Set up parameters
Cs = 1800000 # solar construction cost ($/MW)
Cw = 1400000 # solar construction cost ($/MW), resources: Average U.S. construction costs for solar generation continued to fall in 2019 https://www.eia.gov/todayinenergy/detail.php?id=48736
Vmax = 7.40e+10 # maximun storage of GERD (m^3)
QSmax = 5.30e+7 # maximun spilt (m^3/d)
QOmax = 2.00e+7 # maximun outflow for generation (m^3/d)
Head = 133 # head drop of GERD (m)
Eta = 4650/QOmax # efficiency of GERD (MW/m^3)

## Set up constants
Dh =24 # 24h/day
Hs = 3600 # 3600s/h
Lb = 8000 # base load in 2030, unit MW, reference: https://www.iea.org/articles/ethiopia-energy-outlook
Sa = Vmax/Head # surface area of GERD
Rho = 1000 # density of water kg/m^3
Day = [31,28,31,30,31,30,31,31,30,31,30,31] # Days in months
Lmbd2 = 1200 # loss of load penalty

## Set up data for different scenarios
if case == 0:
    print('running one week simulation...')
    # one week - hourly power, daily flow
    Pr = 168 # power resolution, # of hour in one week
    Qi = Hs*inflow.loc[0:Pr-1].to_numpy() # unit m^3/h
    Qp = Hs*Sa/Rho*prec.loc[0:Pr-1].to_numpy() # precipitation, unit m^3/h
    Qe = Hs*Sa/Rho*evap.loc[0:Pr-1].to_numpy() # evaporation, unit m^3/h
    AvgQ = Dh*Hs*np.mean(inflow_full.loc[0]) # Average of historical outflow
    Lt = Lb*load.loc[0:Pr-1].to_numpy() # unit MW
    CFs = solar.loc[0:Pr-1].to_numpy()
    CFw = wind.loc[0:Pr-1].to_numpy()
    Fr = 7 # flow requirement resolution, # of day in one week
    M = np.zeros([Fr,Pr])
    for i in range(Fr):
        M[i][i*24:i*24+24] = 1
else:
    print('running 5-years simulation...')
    # 5 years - daily power, monthly flow
    Pr = 43800 # power resolution, # of hours in five years
    Qi = Hs*inflow.to_numpy() # unit m^3/h
    Qp = Hs*Sa/Rho*prec.to_numpy() # precipitation, unit m^3/h
    Qe = Hs*Sa/Rho*evap.loc[0:Pr-1].to_numpy() # evaporation, unit m^3/h
    Lt = Lb*load.to_numpy() # unit MW
    CFs = solar.to_numpy()
    CFw = wind.to_numpy()
    Fr = 1825 # flow requirement resolution, # of days in five years
    M = np.zeros([Fr,Pr])
    for i in range(Fr):
        M[i][i*24:i*24+24] = 1
    AvgQY = np.zeros(365)
    for j in range(365):
        AvgQY[j]=Dh*Hs*np.mean(np.mean(inflow_full.loc[j*24:(j+1)*24-1],axis=1))
    AvgQ = np.hstack((AvgQY,AvgQY,AvgQY,AvgQY,AvgQY)).reshape(1825,1)

## Set up variables
gs = cp.Variable(nonneg=True) # solar capacity (MW)
gw = cp.Variable(nonneg=True) # wind capacity (MW)
fl = cp.Variable(shape=(Pr,1), nonneg=True) # loss of load (MW)
ff = cp.Variable(shape=(Fr,1), nonneg=True) # quantity of fail to meet flow requirement (m^3/month)
qo = cp.Variable(shape=(Pr,1), nonneg=True) # quantity of outflow (m^3/h)
qs = cp.Variable(shape=(Pr,1), nonneg=True) # quantity of spilt (m^3/h)
v = cp.Variable(shape=(Pr+1,1), nonneg=False) # reservior storage level (m^3)

## Define the objective function
obj = cp.Minimize(Cs*gs+Cw*gw+cp.sum(Lmbd1*ff)+cp.sum(Lmbd2*fl))

## Define constraints
constraints = [
    Lt <= cp.multiply(Eta,qo) + cp.multiply(CFs,gs) + cp.multiply(CFw,gw) + fl, # load balance
    0.9*AvgQ <= M@qo+ff, #flow requirement
    np.diagflat(np.ones(Pr),1)[0:Pr]@v == np.diagflat(np.ones(Pr),-1)[1:Pr+1]@v + Qi + Qp - Qe - qo - qs, # storage evolution
    v <= Vmax,
    v[0] == 0.75*Vmax,
    v[-1] == 0.75*Vmax,
    qo<=QOmax,
    qs<=QSmax,
]

## Solve the problem
prob = cp.Problem(obj, constraints)
prob.solve(solver = 'GUROBI', verbose=True)

## Result output
generation = {'Hydro (MW)':cp.multiply(Eta,qo).value.reshape(Pr,),
            'Solar (MW)':cp.multiply(CFs,gs).value.reshape(Pr,),
            'Wind (MW)':cp.multiply(CFw,gw).value.reshape(Pr,),
            'Loss Load (MW)':fl.value.reshape(Pr,),
            'Load (MW)':Lt.reshape(Pr,)}
gf = pd.DataFrame(data=generation)
gf.to_csv('generation.csv',header=True)

operation ={'storage (m3)':v.value[1:Pr+1].reshape(Pr,),
            'inflow (m3/h)':Qi.reshape(Pr,),
            'precipitaion (m3/h)':Qp.reshape(Pr,),
            'evaporation (m3/h)':Qe.reshape(Pr,),
            'outflow (m3/h)':qo.value.reshape(Pr,),
            'spilt (m3/h)':qs.value.reshape(Pr,)}
of = pd.DataFrame(data=operation)
of.to_csv('operation.csv',header=True)

result = {'Solar (MW)':[gs.value],
            'Wind (MW)':[gw.value],
            'flow violation (m3)':[cp.sum(ff).value],
            'flow penalty ($)':[cp.sum(Lmbd1*ff).value],
            'loss of load penalty ($)':[cp.sum(Lmbd2*fl).value],
            'Cost ($)':[obj.value]}
rf = pd.DataFrame(data=result)
rf.to_csv('result.csv',header=True)

np.savetxt("flow_violations.csv",ff.value.reshape(Fr,),delimiter=",")
np.savetxt("load_dual.csv",constraints[0].dual_value.reshape(Pr,),delimiter=",")
np.savetxt("flow_dual.csv",constraints[1].dual_value.reshape(Fr,),delimiter=",")
