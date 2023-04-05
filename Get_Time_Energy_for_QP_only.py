import numpy as np
import pandas as pd
import random
import sys, getopt
sys.path.append("/home/kumarpat/Helium_QP_Sim")
from Functions import *
from scipy.interpolate import interp1d
from sklearn.metrics import auc

#Radius_CPD= 38
#Radius_Helium= 30
#Midpoint=-11.645
#Height_CPD=19.2-Midpoint
#half_height_Helium=2.0

#Radius_CPD= 38
#Radius_Helium= 30
#Midpoint=-7.645
#Height_CPD=19.2-Midpoint
#half_height_Helium=6.0


Radius_CPD= 38
Radius_Helium= 30
Midpoint=-5.645
Height_CPD=19.2-Midpoint
half_height_Helium=8.0

#Radius_CPD= 38
#Radius_Helium= 30
#Midpoint=-1.645
#Height_CPD=19.2-Midpoint
#half_height_Helium=12.0


#Radius_CPD= 38
#Radius_Helium= 30
#Midpoint=0.11
#Height_CPD=19.2-Midpoint
#half_height_Helium=13.75

##this one needs to prepare in advance using Generate Dataframe

#Helium = pd.read_pickle('/home/kumarpat/Helium_QP_Sim/4mm/Helium_0.pkl')
#Helium = pd.read_pickle('/home/kumarpat/Helium_QP_Sim/12mm/Helium_0.pkl')
Helium = pd.read_pickle('/home/kumarpat/Helium_QP_Sim/16mm/Helium_0.pkl')
#Helium = pd.read_pickle('/home/kumarpat/Helium_QP_Sim/24mm/Helium_0.pkl')
#Helium = pd.read_pickle('/home/kumarpat/Helium_QP_Sim/27point5mm/Helium_0.pkl')


Collected_Energy=np.zeros(Helium.shape[0])
Prompt_scintillation=np.zeros(Helium.shape[0])
Time_collection=np.zeros(Helium.shape[0])
Time_collection_mode=np.zeros(Helium.shape[0])
X_ini=np.zeros(Helium.shape[0])
Y_ini=np.zeros(Helium.shape[0])
Z_ini=np.zeros(Helium.shape[0])
S2=np.zeros(Helium.shape[0])
E=np.arange(0,1000,10)



omega = Get_P()

domega = Get_V()

p= np.linspace(0.001, 4700, num=470000)#eV
Temp_QP=2
functional_form=p**2*Boltzman_factor_without_V(p,Temp_QP,omega)
normalisation= np.trapz(functional_form, p, p[1]-p[0])

#Loop over the events
for i in range(Helium.shape[0]):
    if(i%100==0):
        print(i)
    #we get energy, momentum magnitude, velocity magnitude etc from geant
    #change the function_form, to change the intial QP populations distribution
    Energy,Momentum,Velocity,X1,Y1,Z1,Intial_X,Intial_Y,Intial_Z=Get_Energy_Velocity_Momentum_Position_df_from_recoil\
                                                       (Helium['quasiparticle_E(eV)'][i],\
                                                        Helium[b'X_He_electron'][i],\
                                                        Helium[b'Y_He_electron'][i],\
                                                        Helium[b'Z_He_electron'][i],
                                                        p ,normalisation, functional_form,omega,domega)



    #get random momentum vectors
    u = np.random.uniform(low=0, high=1,size=np.size(X1))
    theta=2*np.pi*u
    phi= np.arccos(1-2*u)

    Vx_unit_vector=np.sin(phi)*np.cos(theta)
    Vy_unit_vector=np.sin(phi)*np.sin(theta)
    Vz_unit_vector=np.cos(phi)

    #get the prompt energy, no reflection
    Prompt = Get_PromptEnergy_from_recoil_position( Helium['singlet_photon_mean'][i],\
                                                   Helium['IR_photon_mean'][i],\
                                               Intial_X, Intial_Y, Intial_Z, \
                                               Radius_Helium, Radius_CPD,Height_CPD)


    #remove the QP that aere moving downwards
    X,Y,Z,nx, ny, nz,Energy,Momentum,Velocity=Remove_the_downward_QP(Vx_unit_vector,Vy_unit_vector,\
                                                 Vz_unit_vector,  X1, Y1, Z1,\
                                                                          Energy,Momentum,Velocity)
    #get coordinate where the QP hit the He surface
    X_He,Y_He,Z_He,nx_He, ny_He, nz_He,Energy,Momentum,Velocity=get_Coordinate_He_Surface(X,Y,Z,nx, ny, nz,\
                                                                    Energy,Momentum,Velocity,half_height_Helium,Radius_Helium)


    theta_I = Get_Angle_of_incidence(nz_He)
    critical_Angle_of_incidence = Get_critical_Angle_of_incidence(Energy,Velocity)

    #Remove the QP that are not at a favourable angle for evaporation
    X_He_S,Y_He_S,Z_He_S,nx_He_S, ny_He_S, \
                     nz_He_S,Energy,Momentum,Velocity,theta_I_He_S=\
                         Remove_bad_incident_QP(X_He,Y_He,Z_He,nx_He,\
                                                ny_He, nz_He,Energy,Momentum,Velocity,theta_I,critical_Angle_of_incidence)

    # after Quantumn evaporation, you need to chnage the momentum direction
    nx_V,ny_V,nz_V,X_He_S,Y_He_S,Z_He_S,Energy,Velocity,Momentum=\
                                  change_direction(nx_He_S,ny_He_S,nz_He_S,\
                                          X_He_S,Y_He_S,Z_He_S,theta_I_He_S,Energy,Velocity,Momentum)

    # get coordinate where the QP hit the CPD surface
    X_CPD,Y_CPD,Z_CPD,X_He_S,Y_He_S,Z_He_S\
                ,Energy,Momentum,Velocity,Velocity_atom = get_Coordinate_CPD_Surface(X_He_S\
                                 ,Y_He_S,Z_He_S,nx_V,ny_V,nz_V,\
                                     Energy,Momentum,Velocity, Height_CPD,Radius_Helium,Radius_CPD)

    # Calculate the Time it takes for each QP hititng the CPD surface
    Energy_final,total_time_microseconds,position_intx,\
                    position_inty, position_intz=get_Time_QP(X_He_S,Y_He_S,Z_He_S,X_CPD,Y_CPD,Z_CPD,Velocity,\
                                                 Velocity_atom,Energy,Momentum,Intial_X,Intial_Y,Intial_Z)

    Collected_Energy[i]= np.size(Energy_final)
    Prompt_scintillation[i]=Prompt
    Time_collection[i]=total_time_microseconds.mean()
    n,bins=np.histogram(total_time_microseconds, bins=np.arange(0,1000,10))
    Time_collection_mode[i]= bins[np.argmax(n)]
    X_ini[i]=position_intx
    Y_ini[i]=position_inty
    Z_ini[i]=position_intz
    S2[i]=auc( 0.5*(bins[1:] + bins[:-1]), n)
    All_pulses=np.vstack((All_pulses,n))

dataset = pd.DataFrame()
dataset['Evaporation_energy'] = Collected_Energy
dataset['Prompt_energy'] = Prompt_scintillation
dataset['Delay_time'] = Time_collection_mode
dataset['X'] = X_ini
dataset['Y'] = Y_ini
dataset['Z'] = Z_ini
dataset['S2'] = S2
dataset.to_pickle("DTFridge_0.pkl")
np.save("AllPulses_traces",All_pulses)
#dataset.to_hdf('DTFridge_0.h5', key='dataset', mode='w')
