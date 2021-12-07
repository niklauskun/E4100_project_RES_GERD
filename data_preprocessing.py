import pandas as pd

## load data
inflow = pd.read_excel('REVUB_Ethiopia_inflows.xlsx','GERD',header=None).iloc[0:8760,21:26].stack().swaplevel().sort_index() # 2009-2013 data,unit m^3/s, hourly
inflow.to_csv('inflow.csv', index=False, header=None)

inflow_full = pd.read_excel('REVUB_Ethiopia_inflows.xlsx','GERD',header=None).iloc[0:8760,:]
inflow_full.to_csv('inflow_full.csv', index=False, header=None)

evap = pd.read_excel('REVUB_Ethiopia_evaporation.xlsx','GERD',header=None).iloc[0:8760,21:26].stack().swaplevel().sort_index() # 2009-2013 data,unit kg/m^2/s, hourly
evap.to_csv('evap.csv', index=False, header=None)

prec = pd.read_excel('REVUB_Ethiopia_precipitation.xlsx','GERD',header=None).iloc[0:8760,21:26].stack().swaplevel().sort_index() # 2009-2013 data,unit kg/m^2/s, hourly
prec.to_csv('prec.csv', index=False, header=None)

load = pd.read_excel('REVUB_Ethiopia_load.xlsx','Load (2030)',header=None).iloc[0:8760,21:26].stack().swaplevel().sort_index() # 2009-2013 data, hourly
load.to_csv('load.csv', index=False, header=None)

solar = pd.read_excel('REVUB_Ethiopia_solar_CF.xlsx','solar CF (Ethiopia)',header=None).iloc[0:8760,21:26].stack().swaplevel().sort_index() # 2009-2013 data, hourly
solar.to_csv('solar.csv', index=False, header=None)

wind = pd.read_excel('REVUB_Ethiopia_wind_CF.xlsx','wind CF (Ethiopia)',header=None).iloc[0:8760,21:26].stack().swaplevel().sort_index() # 2009-2013 data,hourly
wind.to_csv('wind.csv', index=False, header=None)
