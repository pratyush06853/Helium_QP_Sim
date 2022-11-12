import uproot
import numpy as np
import pandas as pd
import random
import Functions
from scipy.interpolate import interp1d


Singlet= pd.read_table("Singlet.csv", sep=",", usecols=['Energy', 'Partition'])
Triplet = pd.read_table("Triplet.csv", sep=",", usecols=['Energy', 'Partition'])
QP = pd.read_table("QP.csv", sep=",", usecols=['Energy', 'Partition'])
Infrared = pd.read_table("Infrared.csv", sep=",", usecols=['Energy', 'Partition'])



Energy = np.linspace(0, 10, num=100000)
#f is singlet, g is triplet as the paper had it reversed

g = interp1d(Singlet["Energy"]/1000, Singlet["Partition"],fill_value=(0,  Singlet["Partition"][ len( Singlet["Partition"])-1] ), bounds_error=False)
SingletPartition= g(Energy)

f = interp1d(Triplet["Energy"]/1000, Triplet["Partition"],fill_value=(0, Triplet["Partition"][ len( Triplet["Partition"])-1]), bounds_error=False)
TripletPartition= f(Energy)

h = interp1d(QP["Energy"]/1000, QP["Partition"],fill_value=(0, QP["Partition"][ len( QP["Partition"])-1] ), bounds_error=False)
QPPartition = h(Energy)

IR = interp1d(Infrared["Energy"]/1000, Infrared["Partition"],fill_value=(0, QP["Partition"][ len( QP["Partition"])-1] ), bounds_error=False)
QPPartition = IR(Energy)



Radius_CPD= 38
Radius_Helium= 30
Midpoint=-11.645
Height_CPD=19.2-Midpoint
half_height_Helium=2.0

#Radius_CPD= 38
#Radius_Helium= 30
#Midpoint=-7.645
#Height_CPD=19.2-Midpoint
#half_height_Helium=6.0


#Radius_CPD= 38
#Radius_Helium= 30
#Midpoint=-5.645
#Height_CPD=19.2-Midpoint
#half_height_Helium=8.0

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



tree =uproot.open("/Volumes/GoogleDrive/My Drive/GraduateWork/HeRALD/raw_geant_outputdir/ProcessedFiles/Run30/4mm/Combined4mm_ForHeSim.root")
#tree =uproot.open("/Volumes/GoogleDrive/My Drive/GraduateWork/HeRALD/raw_geant_outputdir/ProcessedFiles/Run30/12mm/Combined12mm_ForHeSim.root")
#tree =uproot.open("/Volumes/GoogleDrive/My Drive/GraduateWork/HeRALD/raw_geant_outputdir/ProcessedFiles/Run30/16mm/Combined27point5mm_ForHeSim.root")
#tree =uproot.open("/Volumes/GoogleDrive/My Drive/GraduateWork/HeRALD/raw_geant_outputdir/ProcessedFiles/Run30/24mm/Combined27point5mm_ForHeSim.root")
#tree =uproot.open("/Volumes/GoogleDrive/My Drive/GraduateWork/HeRALD/raw_geant_outputdir/ProcessedFiles/Run30/27point5mm/Combined27point5mm_ForHeSim.root")



tree3=tree["Event"]
arr3 = tree3.arrays(tree3.keys())
df_uproot3 = pd.DataFrame(arr3)

t =tree["Information"]
a = t.arrays(t.keys())

##Select only Helium things
Helium = df_uproot3[[b'E_He_electron',b'X_He_electron',b'Y_He_electron',b'Z_He_electron',b'TotalE_He',b'TotalE_CPD']]
Helium['Total_E'] = Helium[b'TotalE_He'] +Helium[b'TotalE_CPD']
Helium= Helium[ (Helium[b'TotalE_He']!=0) ]

Helium= Helium[ (Helium[b'TotalE_He']>=5.8) ]
Helium= Helium[ (Helium[b'TotalE_He']<=7.5) ]

Helium[b'Z_He_electron']=Helium[b'Z_He_electron']-Midpoint
Helium[b'Radius']=np.sqrt(Helium[b'X_He_electron']**2+Helium[b'Y_He_electron']**2)

##energy partition
singlet_fraction,singlet_photon_mean,\
IR_fraction, IR_photon_mean,quasiparticle_fraction,triplet_photon_mean =Functions.Energy_partion(np.asarray( Helium[b'TotalE_He'] ),f,g,IR)

Helium['quasiparticle_E(eV)'] = quasiparticle_fraction*Helium[b'TotalE_He']*1e3

Helium['singlet_photon_mean'] = singlet_photon_mean
Helium['triplet_photon_mean'] = triplet_photon_mean
Helium['IR_photon_mean'] = IR_photon_mean

#Helium = Helium.sample(n=1000)
Helium.reset_index(inplace=True)
No_of_events_per_file=1000

for i in range(int(Helium.shape[0]/1000)-1):
    Helium[int(i*No_of_events_per_file):int((i+1)*No_of_events_per_file)].to_pickle("4mm/Helium_"+ str(i)+".pkl")
