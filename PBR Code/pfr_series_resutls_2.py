import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime


def master_function(fun,tspan, y0, method='rk4', number_of_points=100):
    '''General function to solve system of differential equations. Does not work on single differential equations. \n
    fun = function 
    y0 = vector of initial conditions
    optional:\n
    method = You can select the method with which your system of differential equations will be evaluated. Default set to second order Runge-Kutta. \n
    Supported methods : midpoint method ('midpoint'), euler method ('euler'), Classical second order Runge-Kutta ('rk2'), classical fourth order Runge-Kutta ('rk4').
    number_of_points = how many steps. Default set to 100. Increasing this reduces error but increases computation time. '''
    dt = (tspan[1] - tspan[0])/number_of_points
    t = np.linspace(tspan[0], tspan[1], number_of_points+1)
    y = np.zeros((number_of_points+1, len(y0))) # len(y0) because you would need an initial condition for each derivative.
    for i in range(len(y0)): #initial conditions as a loop to ensure universability.
        y[0,i] = y0[i]
    if method == 'midpoint':
        for i in range(number_of_points):
            k1 = fun(t[i], y[i,:])
            k2 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k1)
            y[i+1,:] = y[i,:] + dt * k2
    elif method == 'euler':
        for i in range(number_of_points):
            y[i+1,:] = y[i,:] + dt * fun(t[i], y[i,:])
    elif method == 'rk2':
        for i in range(number_of_points):
            k1 = fun(t[i], y[i,:])
            k2 = fun(t[i] + dt, y[i] + dt*k1)
            y[i+1,:] = y[i] + dt*0.5*(k1+k2)
    elif method == 'rk4':
        for i in range(number_of_points):
            k1 = fun(t[i], y[i,:])
            k2 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k1)
            k3 = fun(t[i] + dt*0.5, y[i,:] + 0.5*dt*k2)
            k4 = fun(t[i] +dt, y[i,:] + dt*k3)
            y[i+1,:] = y[i] + dt*((1/6)*k1 + (1/3)*(k2+k3) + (1/6)*k4)
    else:
        return 'Unknown method specified. Check documentation for supported methods' # In case an unknown method is specified
    return t, y

def CSTR_series_model(T,fv1,fv2, V=137, tspan = [0,3600]):
    '''Models the behavior of the reaction: Water + Acetic Anhydride -> 2 * Acetic acid in an adiabatic CSTR reactor. \n
    Required Arguments: \n
    T = inlet temperature for the reactor given in units celsius \n
    fv1 = flow rate of water in units ml/min \n
    fv2 = flow rate of acetic anhydride ml/min \n
    Optional Arguments: \n
    V = volume of the reactor in units ml (default set to 500ml) \n
    tspan = list of evaluation time in units seconds (default set to [0,3600]) \n
    This function was built for the course "Practical Process Technology (6P4X0)" 
    '''
    v_pfr= V/8 #devide the pfr volume by the number of reactors

    # Convert flow rates (ml/min to ml/s)
    fv_w_dm3_s = fv1 / 60  # Water flow rate in ml/s
    fv_a_dm3_s = fv2  / 60  # Anhydride flow rate in ml/s


    
    #Chemical constants
    mm_water = 18.01528 # (g/mol)
    rho_water = 0.999842 # (g/ml)
    cw_pure = rho_water/mm_water # (mol/ml)

    mm_AAH = 102.089 # (g/mol)
    rho_AAH = 1.082 # (g/ml)
    caah_pure = rho_AAH/mm_AAH # (mol/ml)

    flow_array = [fv_w_dm3_s, fv_a_dm3_s]

    #Thermodynamic constants
    params = {
        "C_in_water": (flow_array[0]*cw_pure)/(flow_array[0]+flow_array[1]),
        "C_in_AAH": (flow_array[1]*caah_pure)/(flow_array[0]+flow_array[1]),
        "Inlet temperature": T+273.15,
        "flow": flow_array,
        "V": v_pfr,  # Volume in ml
        "k0": 6e5,          # Reaction rate constant (ml/mol/s)

        # Thermodynamic constants (taken from Asprey et al., 1996)
        "Ea": 45622.34,             # Activation energy (J/mol)
        "R": 8.314,              # Gas constant (J/mol/K)
        "H": -56.6e3,              # Enthalpy change (J/mol)
        "rho": 1,            # Density (g/ml)
        "cp": 4.186             # Heat capacity (J/g/K)
    }
    xini = [cw_pure,0,0,T+273.15] # Initial Conditions 


    sol_tank1 = master_function(lambda t, C: der_func_first_reactor(t, C, params), tspan, xini, method='rk4', number_of_points=300)

    # Solves for reactors 2 to 8 
    sol_tanks = [sol_tank1]
    for tank_num in range(1, 8):
        sol_tank_prev = sol_tanks[-1][1]  #copies the whole last take solution
        sol_tank_current = np.zeros_like(sol_tank_prev)  #prepares an array of zeros which will later be replaced by values 
        
        for i, t in enumerate(sol_tanks[-1][0]):
            c_old = sol_tank_prev[i, :]  #previous tanks solution
            sol_tank = master_function(lambda t, C: der_func_other_reactors(t, C, params, c_old), [t, t + (tspan[1] - tspan[0]) / 100], c_old, method='rk4', number_of_points=1)
            sol_tank_current[i, :] = sol_tank[1][-1, :]  

        sol_tanks.append((sol_tanks[-1][0], sol_tank_current))  # Appends the current tank's solution 

    return sol_tanks

def der_func_first_reactor(t,C, parameters):
    # Initializing derivative vector
    dcdt = np.zeros(4)

    # Getting parameters
    C_in_w = parameters['C_in_water']
    C_in_AAH = parameters['C_in_AAH']
    flow = parameters['flow']
    V = parameters['V']
    k0 = parameters['k0']
    Ea = parameters['Ea']
    R = parameters['R']
    H = parameters['H']
    rho = parameters['rho']
    cp = parameters['cp']
    inlet_temp = parameters["Inlet temperature"]
    
    reaction_rate = C[0]*C[1] * k0 * np.exp(-Ea/(R*C[3])) 

    total_flow = flow[0]+flow[1]
    #Differential equations
    dcdt[0] =  (total_flow/V)*(C_in_w - C[0])    - reaction_rate # reaction_rate # Water Concentration derv
    dcdt[1] =  (total_flow/V)*(C_in_AAH - C[1])  - reaction_rate  # Anhydride Concentration derv
    dcdt[2] =  (total_flow/V)*(0 - C[2]) + 2*reaction_rate # 2*reaction_rate # Acetic acid 
    dcdt[3] =  (total_flow/V) * (inlet_temp-C[3]) - H/(rho*cp) * reaction_rate # Temperature part
    return dcdt

def der_func_other_reactors(t,C, parameters, c_old):
    # Initializing derivative vector
    dcdt = np.zeros(4)

    # Getting parameters
    C_in_w = parameters['C_in_water']
    C_in_AAH = parameters['C_in_AAH']
    flow = parameters['flow']
    V = parameters['V']
    k0 = parameters['k0']
    Ea = parameters['Ea']
    R = parameters['R']
    H = parameters['H']
    rho = parameters['rho']
    cp = parameters['cp']
    inlet_temp = parameters["Inlet temperature"]
    
    reaction_rate = C[0]*C[1] * k0 * np.exp(-Ea/(R*C[3])) 

    total_flow = flow[0]+flow[1]
    #Differential equations
    dcdt[0] =  (total_flow/V)*(c_old[0] - C[0]) - reaction_rate # reaction_rate # Water Concentration derv
    dcdt[1] =  (total_flow/V)*(c_old[1] - C[1])  - reaction_rate  # Anhydride Concentration derv
    dcdt[2] =  (total_flow/V)*(c_old[2] - C[2]) + 2*reaction_rate # 2*reaction_rate # Acetic acid 
    dcdt[3] =  (total_flow/V) * (c_old[3]-C[3]) - H/(rho*cp) * reaction_rate # Temperature part
    return dcdt

def temp_extract(data, x, offset=0):
    # Extract the flow data to determine the starting time
    flow_rows = data[data['TagName'] == "P120_Flow"]
    valid_flow_rows = [row for row in flow_rows if row['vValue'] not in ['(null)', None]]
    flow_values = [float(row['vValue']) for row in valid_flow_rows]
    flow_dates = [datetime.strptime(row['DateTime'].split('.')[0], '%Y-%m-%d %H:%M:%S') for row in valid_flow_rows]

    start_time = None
    for i in range(1, len(flow_values)):
        if flow_values[i-1] < 1 and flow_values[i] > 1:
            start_time = flow_dates[i]
            break
    
    # Extract temperature data starting from the transition point
    temp_rows = data[data['TagName'] == x]
    valid_temp_rows = [row for row in temp_rows if row['vValue'] not in ['(null)', None]]
    
    temp_dates = [datetime.strptime(row['DateTime'].split('.')[0], '%Y-%m-%d %H:%M:%S') for row in valid_temp_rows]
    temp_values = [float(row['vValue']) + offset for row in valid_temp_rows]

    elapsed_time = [(dt - start_time).total_seconds() / 60 for dt in temp_dates] # Calculate elapsed time in minutes from the start_time

    return elapsed_time, temp_values, (flow_dates[0] - start_time).total_seconds() / 60 


my_data = np.genfromtxt('Data/PFR/25.09.30C.csv', delimiter=';', dtype=None, names=True, encoding='ISO-8859-1') #imports the data and saves it as variable

# extracting all temp data
t_values = ['T208_PV','T207_PV','T206_PV','T205_PV','T204_PV','T203_PV','T202_PV','T201_PV','T200_PV']
results = {}
for t_value in t_values:
    elap_time, temp_c, offset_time = temp_extract(my_data, t_value)
    results[t_value] = {'elapsed_time': elap_time, 'temperature': temp_c, 'offset_time':offset_time}

#Get AAH Flowrate
elapsed_time_c_aah, aah_flowrate_c_vector, offset_time = temp_extract(my_data, x="P120_Flow")

#Get Water Flowrate
elapsed_time_c_water, water_flowrate_c_vector, offset_time = temp_extract(my_data, x='P100_Flow')

initial_temperature = np.min(temp_c)
aah_flowrate_c = np.median(aah_flowrate_c_vector)
water_flowrate_c = np.median(water_flowrate_c_vector)

sol_me = CSTR_series_model(initial_temperature, water_flowrate_c, aah_flowrate_c, V=137) 

fig, ax = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=True)
ax = ax.flatten()
for i in range(0,8):
    ax[i].plot(results[t_values[-(i+1)]]['elapsed_time'], np.array(results[t_values[-(i+1)]]['temperature']) - results[t_values[-(i+1)]]['temperature'][0],color='#ff7f0e',label= 'real')
    ax[i].plot(np.array(results[t_values[-(i+1)]]['elapsed_time']) - results[t_values[-(i+1)]]['offset_time'], sol_me[i][1][:101, 3] - sol_me[i][1][0, 3], color='#1f77b4',label = 'model')

    ax[i].set_title(f'reactor {i + 1} Data')
    ax[i].set_xlabel('Elapsed Time (min)')
    ax[i].set_ylabel('Temperature (°C)')
    ax[i].set_xlim(0,15)
    ax[i].grid(True)
    ax[i].set_xlim(0,15)
    ax[i].legend()
fig.suptitle('Temperature = 30C', fontsize=16) # Adjust layout to prevent label overlap and set a global title
plt.tight_layout(rect=[0, 0, 1, 0.95])  # fixes overlap
plt.show()


