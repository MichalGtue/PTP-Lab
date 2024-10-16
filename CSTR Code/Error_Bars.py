import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.integrate
# Assume reaction is 1st order wrt both components
# Assume isothermal (no exotherm)
# Assume constant density


def CSTR_model(T,fv1,fv2, V=500, tspan = [0,3600]):
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
    v_cstr=V

    # Convert flow rates (ml/min to ml/s)
    fv_w_dm3_s = fv1 / 60  # Water flow rate in ml/s
    fv_a_dm3_s = fv2  / 60  # Anhydride flow rate in ml/s

    #Chemical constants
    #Water
    mm_water = 18.01528 # (g/mol)
    rho_water = 0.999842 # (g/ml)
    cw_pure = rho_water/mm_water # (mol/ml)
    #Acetic acid
    mm_AAH = 102.089 # (g/mol)
    rho_AAH = 1.082 # (g/ml)
    caah_pure = rho_AAH/mm_AAH # (mol/ml)

    flow_array = [fv_w_dm3_s, fv_a_dm3_s]
    
    params = { # Stores the relevant thermodynamic constants as a dictionary 
        "C_in_water": (flow_array[0]*cw_pure)/(flow_array[0]+flow_array[1]),
        "C_in_AAH": (flow_array[1]*caah_pure)/(flow_array[0]+flow_array[1]),
        "Inlet temperature": T+273.15, # Temp but now in kelvin
        "flow": flow_array,
        "V": v_cstr,  # Volume in ml
        "k0": 7e6,          # Reaction rate constant (ml/mol/s)

        # Thermodynamic constants (taken from Asprey et al., 1996)
        "Ea": 45622.34,             # Activation energy (J/mol)
        "R": 8.314,              # Gas constant (J/mol/K)
        "H": -56.6e3,              # Enthalpy change (J/mol)
        "rho": 1,            # Density (g/ml)
        "cp": 4.186             # Heat capacity (J/g/K)
    }
    # print(params['C_in_AAH']*params['C_in_water'])
    xini = [cw_pure,0,0,T+273.15] # Initial Conditions 
    sol_me = scipy.integrate.solve_ivp(der_func, tspan, xini, args=(params,))
    return sol_me

def der_func(t,C, parameters):
    '''This function contains the differential equations to solve the reaction A+B->2C in an adiabatic 
    CSTR. \n
    t=time (seconds) \n
    c = Concentration vector like [c_water, c_AAH, c_AA, Temperature]\n
    parameters = dictionary containing thermodynamic constants
    '''
    # Initializing derivative vector
    dcdt = np.zeros(4)
    # array of 4 zeros corresponding to [c_water/dt, c_AAH/dt, c_AA/dt, dT/dt]

    # Getting parameters out of our dictionary 
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
    
    reaction_rate = C[0]*C[1] * k0 * np.exp(-Ea/(R*C[3])) # reaction rate is repeated so just calculate once

    total_flow = flow[0]+flow[1]
    
    #Differential equations
    dcdt[0] =  (total_flow/V)*(C_in_w - C[0])    - reaction_rate # Water Concentration derv
    dcdt[1] =  (total_flow/V)*(C_in_AAH - C[1])  - reaction_rate  # Anhydride Concentration derv
    dcdt[2] =  (total_flow/V)*(0 - C[2]) + 2*reaction_rate  # Acetic acid 
    dcdt[3] =  (total_flow/V) * (inlet_temp-C[3]) - H/(rho*cp) * reaction_rate # Temperature part
    return dcdt

def temp_extract(data, x="T200_PV", offset=0):
    '''Function to extract data from csv files\n
    data = data path for your csv file. Give as a string \n
    x = Name of the instrument that you want. Default set to T200_PV (CSTR internal temperature) \n
    offset = linear offset for values. Default set to zero \n
    returns elapsed time and values for your
    '''
    # Extract the flow data to determine the starting time
    flow_rows = data[data['TagName'] == "P120_Flow"]
    valid_flow_rows = [row for row in flow_rows if row['vValue'] not in ['(null)', None]]
    flow_values = [float(row['vValue']) for row in valid_flow_rows]
    flow_dates = [datetime.strptime(row['DateTime'].split('.')[0], '%Y-%m-%d %H:%M:%S') for row in valid_flow_rows]
    start_time = None
    for i in range(1, len(flow_values)):
        if flow_values[i-1] < 1 and flow_values[i] > 1:     # Loop that checks when the AAH pump is turned on and sets that as the start time
            start_time = flow_dates[i]
            break # Stop the loop once flow starts

    temp_rows = data[data['TagName'] == x]  # Only choose the rows for that particular instrument 
    valid_temp_rows = [row for row in temp_rows if row['vValue'] not in ['(null)', None]] # You want to remove the values when theres null otherwise it does weird things
    
    temp_dates = [datetime.strptime(row['DateTime'].split('.')[0], '%Y-%m-%d %H:%M:%S') for row in valid_temp_rows] #Converts the weird csv time format to python
    temp_values = [float(row['vValue']) + offset for row in valid_temp_rows]

    # Calculate elapsed time in minutes from the start_time
    elapsed_time = [(dt - start_time).total_seconds() / 60 for dt in temp_dates]

    return elapsed_time, temp_values

def data_extract(data_path):
    '''Extracts the initial conditions for a the reaction \n
    Data_Path = relative path to the csv document'''
    data_numpy = np.genfromtxt(data_path, delimiter=';', dtype=None, names=True, encoding=None) #built in numpy function to extract data

    #Get temperature
    elapsed_time, temp = temp_extract(data_numpy) 

    #Get AAH Flowrate
    elapsed_time_aah, aah_flowrate_vector = temp_extract(data_numpy, x="P120_Flow")

    #Get Water Flowrate
    elapsed_time_water, water_flowrate_vector = temp_extract(data_numpy, x='P100_Flow')

    initial_temperature = np.min(temp) # Minimum temp = ini temp
    aah_flowrate = np.median(aah_flowrate_vector) # better than the average because sometimes we press prime before the experiment starts
    water_flowrate = np.median(water_flowrate_vector) # the signal is also kinda noisy 
    return elapsed_time, temp, initial_temperature, aah_flowrate, water_flowrate


if __name__ == '__main__':
    # Load the datasets
    data_22c = data_extract('Data\\CSTR\\Runs 16.09\\CSTR 27c.csv')
    data_round2 = data_extract('Data\\Data from trade\\CSTR\\experiment14.10.csv')
    
    # Convert to numpy arrays
    elapsed_time_22c, temp_22c, initial_temperature_22c, aah_flowrate_22c, water_flowrate_22c = data_22c
    elapsed_time_round2, temp_round2, initial_temperature_round2, aah_flowrate_round2, water_flowrate_round2 = data_round2

    # Select specific slices
    time_slice_22c = elapsed_time_22c[4:80]  # 4 to 80
    temp_slice_22c = temp_22c[4:80]

    time_slice_round2 = elapsed_time_round2[12:49]  # 12 to 49
    temp_slice_round2 = temp_round2[12:49]

    # Create an array of times and temperatures
    # Since the time ranges may not match, we will align the datasets
    common_time = np.union1d(time_slice_22c, time_slice_round2)  # Get all unique time points

    # Interpolate temperature values for the common time points
    temp_interpolated_22c = np.interp(common_time, time_slice_22c, temp_slice_22c)
    temp_interpolated_round2 = np.interp(common_time, time_slice_round2, temp_slice_round2)

    # Calculate average temperature and standard deviation
    avg_temp = (temp_interpolated_22c + temp_interpolated_round2) / 2
    std_temp = np.std([temp_interpolated_22c, temp_interpolated_round2], axis=0)

    plt.figure(figsize=(10, 6))
    plt.errorbar(common_time, avg_temp, yerr=std_temp, fmt='-o', label='Average Temperature', color='blue', capsize=5, elinewidth=1, markerfacecolor='red', markeredgecolor='black')
    
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title('Average Temperature with Standard Deviation Error Bars')
    plt.legend()
    plt.grid()
    plt.xlim(0, 30)
    plt.show()

    print("Average Temperature:", avg_temp)
    print("Standard Deviation:", std_temp)
