import numpy as np

# Constants, data, parameters...

rho = 1025.0 # units: kg/m^3
c = 4000.0 # units: J/(kg C)

dt = 10.0 * 86400 # units: s
dz = 5.0 # units: m

T_top = 24.0 # Top temperature
T_bot = 6.0 # Bottom temperature

z_mix = 50.0 # Depth of minimum mixing layer
z_base = 1000.0 # Depth of thermocline



### Intermediate functions

def steady_state(z):
    mixing_index = np.where(z_mix - z == 0)[0][0]
    mixing_layer = T_top * np.ones(mixing_index)
    z = z[mixing_index + 1:]
    thermocline = ((T_top - T_bot) * z - (T_top * z_base - T_bot * z_mix)) / (z_mix - z_base)
    return np.concatenate((mixing_layer, thermocline), axis=None)


def convective_adjustment(T):
    
    # The logic of this loop is predicated on there being always being mixing layer.
    
    def mixing_Rahmstorf(T):
        
        for i in range(len(T) - 1):
            
            if T[i] < T[i + 1]: # if there is an instability
                
                j = 0
                
                # This loop mixes the instability downwards
                
                while T[i + j] < T[i + j + 1]:
                    
                    T[i : i + j + 2] = np.mean(T[i : i + j + 2])
                    
                    j += 1
                    
                break
            
        return T
    
    # Run the Rahmstorf mixing algorithm until the instability has been eliminated from the water column (i.e., from the mixing layer) 
    
    for k in range(int(z_mix / dz)):
        mixing_Rahmstorf(T)
    return mixing_Rahmstorf(T)


def OHC(T):
    OHC = np.empty(T.shape[1])
    
    for i in range(len(OHC)):
        OHC[i] = np.mean(T[:, i])
    return OHC



### The forward-in-time, central-in-space scheme with restoring and flux boundary conditions

def FTCS(dt, dz, kappa, gamma, mu, z_mix, years, T_initial, Q, T0):
    
    # Data
    
    radiative_forcing = Q / (c * rho * z_mix)
    
    days = dt / 86400 # Timestep in days
    M = int(z_base / dz) # Number of spatial steps
    N = int(years * 360 / days) # Number of timesteps in days (taking 1 year = 360 days)
    z = np.linspace(0.0, z_base, M + 1)
    t = np.linspace(0.0, years * 360, N + 1)
    
    Hmix = np.heaviside(z_mix - z, 0)
    z_mix_index = np.where(z_mix - z == 0)[0][0] # this is the index of z_mix in z
    
    
    # Temperature and flux. Prescribe initial and boundary conditions
    
    T = np.zeros((M, N + 1))
    F = np.zeros((M + 1, N + 1))
    
    T[:, 0] = T_initial # Initial condition
    #T[-1, :] = T_bot
    F[0, :] = 0 # No flux at surface
    F[- 1, :] = 0 # No flux at base
        
        
    for n in range(0, N):
        
        for m in range(1, M):
            
            F[m, n] = kappa / dz * (T[m - 1, n] - T[m, n])
            
        for m in range(0, M): # NB: Needs to go to M - 1 if setting the temperature at the base.
             
            T[m, n + 1] = T[m, n] + dt / dz * (F[m, n] - F[m + 1, n]) \
            + dt * (radiative_forcing[n] - gamma * (T[m, n] - (T0[n] + T[m, 0]))) * Hmix[m]  \
            - dt * mu * (T[m, n] - T[m, 0]) * np.heaviside(z - 900, 1)[m] 
            
            #  + w * dt * (- F[m + 1, n] / kappa) \
            # NB: "+ w" since z is positive downwards, this is an upwelling velocity, implying that it would have been negative on the LHS in order to balance diffusion

            
        #Convective adjustment step
    
        convective_adjustment(T[:, n + 1])
        
    return T