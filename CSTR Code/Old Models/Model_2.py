import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Assume reaction is 1st order wrt both components
# Assume isothermal (no exotherm)
# Assume constant density

# Flow parameters
fv_w = 161.10420227050781 # Water Flow Rate (ml/min)
fv_a = 11.524953842163086 # Anhydride Flow Rate (ml/min)
v_cstr = 1 # Volume of CSTR (ml)

#Convert flows to units (ml/s)
fv_w_s = fv_w/60
fv_a_s = fv_a/60

#Make an array with the flows
flow_array = [fv_w_s,fv_a_s]

#Thermodynamic constants
params = {
    "flow": flow_array,
    "V": v_cstr,  # Volume in ml
    "k0": 0.02, #1.8725e3,          # Reaction rate constant (ml/mol/s)

    # Thermodynamic constants (taken from Asprey et al., 1996)
    "Ea": 45622.34,             # Activation energy (J/mol)
    "R": 8.314,              # Gas constant (J/mol/K)
    "H": 56.6e3,              # Enthalpy change (J/mol)
    "rho": 1,            # Density (g/ml)
    "cp": 4.186             # Heat capacity (J/g/K)
}





def der_func(t,C, parameters):
    # Initializing derivative vector
    dcdt = np.zeros(4)

    # Getting parameters
    flow = parameters['flow']
    V = parameters['V']
    k0 = parameters['k0']
    Ea = parameters['Ea']
    R = parameters['R']
    H = parameters['H']
    rho = parameters['rho']
    cp = parameters['cp']

    #reaction_rate = C[0]*C[1] * k0  * np.exp(-Ea/R*C[3])
    total_flow = flow[0]+flow[1]

    #Differential equations
    dcdt[0] =  (flow[0]/V)*(2 - C[0])    - k0*C[0]*C[1] # reaction_rate # Water Concentration derv
    dcdt[1] =  (flow[1]/V)*(1 - C[1])    - k0*C[0]*C[1] # reaction_rate # Anhydride Concentration derv
    dcdt[2] =  (total_flow/V)*(0 - C[2]) + 2*k0*C[0]*C[1] # 2*reaction_rate # Acetic acid 
    dcdt[3] =  0#(total_flow/V) * (ini_cond[3]-C[3]) - H/(rho*cp) * reaction_rate # Temperature part
    return dcdt



tspan = [0,60] # Time in seconds

#For xini, c_water_0, c_AAH_0, C_AA_0, T0(in C)
xini = [3,0,0,0] # Initial Conditions 

sol = scipy.integrate.solve_ivp(der_func, tspan, xini, args=(params,))


plt.plot(sol.t, sol.y[2])
plt.show()
# plt.plot(sol.t, sol.y[0], label='Ca')
# plt.plot(sol.t, sol.y[1], label='Cb')
# plt.plot(sol.t, sol.y[2], label='Cc')
# plt.xlabel('time')
# plt.ylabel('Concentration(mol/L)')
# plt.legend()
# plt.title('Concentration of various components in a CSTR')
# plt.show()




