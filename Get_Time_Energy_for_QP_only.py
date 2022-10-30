import uproot
import numpy as np
import pandas as pd
import random
import Functions
from scipy.interpolate import interp1d

Radius_CPD= 38
Radius_Helium= 30
Midpoint=-7.645
Height_CPD=19.2-Midpoint
half_height_Helium=6.0


Helium = pd.read_pickle('12mm/Helium_0.pkl')
Collected_Energy=np.zeros(Helium.shape[0])
Time_collection=np.zeros(Helium.shape[0])
Time_collection_mode=np.zeros(Helium.shape[0])
X_ini=np.zeros(Helium.shape[0])
Y_ini=np.zeros(Helium.shape[0])
Z_ini=np.zeros(Helium.shape[0])



omega = Functions.Get_P()

domega = Functions.Get_V()

p= np.linspace(0.001, 4700, num=470000)#eV
Temp_QP=2
functional_form=p**2*Functions.Boltzman_factor_without_V(p,Temp_QP,omega)
normalisation= np.trapz(functional_form, p, p[1]-p[0])


for i in range(Helium.shape[0]):
    if(i%100==0):
        print(i)
    Energy,Momentum,Velocity,X1,Y1,Z1,Intial_X,Intial_Y,Intial_Z=Functions.Get_Energy_Velocity_Momentum_Position_df_from_recoil\
                                                       (Helium['quasiparticle_E(eV)'][i],\
                                                        Helium[b'X_He_electron'][i],\
                                                        Helium[b'Y_He_electron'][i],\
                                                        Helium[b'Z_He_electron'][i],
                                                        p ,normalisation, functional_form,omega,domega)




    u = np.random.uniform(low=0, high=1,size=np.size(X1))
    theta=2*np.pi*u
    phi= np.arccos(1-2*u)

    Vx_unit_vector=np.sin(phi)*np.cos(theta)
    Vy_unit_vector=np.sin(phi)*np.sin(theta)
    Vz_unit_vector=np.cos(phi)



    X,Y,Z,nx, ny, nz,Energy,Momentum,Velocity=Functions.Remove_the_downward_QP(Vx_unit_vector,Vy_unit_vector,\
                                                 Vz_unit_vector,  X1, Y1, Z1,\
                                                                          Energy,Momentum,Velocity)

    X_He,Y_He,Z_He,nx_He, ny_He, nz_He,Energy,Momentum,Velocity=Functions.get_Coordinate_He_Surface(X,Y,Z,nx, ny, nz,\
                                                                    Energy,Momentum,Velocity,half_height_Helium,Radius_Helium)


    theta_I = Functions.Get_Angle_of_incidence(nz_He)
    critical_Angle_of_incidence = Functions.Get_critical_Angle_of_incidence(Energy,Velocity)


    X_He_S,Y_He_S,Z_He_S,nx_He_S, ny_He_S, \
                     nz_He_S,Energy,Momentum,Velocity,theta_I_He_S=\
                         Functions.Remove_bad_incident_QP(X_He,Y_He,Z_He,nx_He,\
                                                ny_He, nz_He,Energy,Momentum,Velocity,theta_I,critical_Angle_of_incidence)


    nx_V,ny_V,nz_V,X_He_S,Y_He_S,Z_He_S,Energy,Velocity,Momentum=Functions.change_direction(nx_He_S,ny_He_S,nz_He_S,\
                                          X_He_S,Y_He_S,Z_He_S,theta_I_He_S,Energy,Velocity,Momentum)


    X_CPD,Y_CPD,Z_CPD,X_He_S,Y_He_S,Z_He_S\
                ,Energy,Momentum,Velocity,Velocity_atom = Functions.get_Coordinate_CPD_Surface(X_He_S\
                                 ,Y_He_S,Z_He_S,nx_V,ny_V,nz_V,\
                                     Energy,Momentum,Velocity, Height_CPD,Radius_Helium,Radius_CPD)

    Energy_final,total_time_microseconds,position_intx,\
                    position_inty, position_intz=Functions.get_Time_QP(X_He_S,Y_He_S,Z_He_S,X_CPD,Y_CPD,Z_CPD,Velocity,\
                                                 Velocity_atom,Energy,Momentum,Intial_X,Intial_Y,Intial_Z)

    Collected_Energy[i]=(9*1e-3)*np.size(Energy_final)
    Time_collection[i]=total_time_microseconds.mean()
    n,bins=np.histogram(total_time_microseconds, bins=np.arange(0,1000,10))
    Time_collection_mode[i]= bins[np.argmax(n)]
    X_ini[i]=position_intx
    Y_ini[i]=position_inty
    Z_ini[i]=position_intz


np.save('12mm_results/Prompt_channel_12mm_WithIR_WithRnegative_60percent_RyanCorrection_2000mK_longlived_QP.npy',Prompt_scintillation)
np.save('12mm_results/Evaporation_channel_12mm_WithIR_WithRnegative_60percent_RyanCorrection_2000mK_longlived_QP.npy',Collected_Energy)
np.save('12mm_results/Delay_Time_12mm_WithIR_WithRnegative_60percent_RyanCorrection_2000mK_longlived_QP.npy', Time_collection_mode)
np.save('12mm_results/Intial_X_12mm_WithIR_WithRnegative_60percent_RyanCorrection_2000mK_longlived_QP.npy', X_ini)
np.save('12mm_results/Intial_Y_12mm_WithIR_WithRnegative_60percent_RyanCorrection_2000mK_longlived_QP.npy', Y_ini)
np.save('12mm_results/Intial_Z_12mm_WithIR_WithRnegative_60percent_RyanCorrection_2000mK_longlived_QP.npy', Z_ini)
