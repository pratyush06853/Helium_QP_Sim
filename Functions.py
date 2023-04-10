import numpy as np
import pandas as pd
import random
from scipy.interpolate import interp1d

def Get_P(filename):
    '''
    returns f which can be used to map momentum in eV to energy in eV
    '''
    Dispersion_data = pd.read_table(filename, sep=",", usecols=['Momentum', 'Energy'])
    Dispersion_data['Momentum']=Dispersion_data['Momentum']*1e3
    Dispersion_data['Energy']=Dispersion_data['Energy']*1e-3

    #f = interp1d(Dispersion_data['Momentum'],Dispersion_data['Energy'],fill_value=(0, 0), bounds_error=False)


    omega=np.polynomial.polynomial.Polynomial.fit(Dispersion_data['Momentum'], Dispersion_data['Energy'], 7, window=[0, 4900] )
    omega_params =omega.convert().coef

    #return f
    return omega_params


def Omega(p,omega_params):
    '''
    returns f which can be used to map momentum in eV to energy in eV
    '''
    e=np.zeros(np.size(p))
    for i in range(len(omega_params)):
        e += omega_params[i]*pow(p, i)
    return e #returns energy in eV


def Get_NoOfQuasiparticles(qp_energy):
    #returns the mean number of quasiparticles given
    # the energy in eV in the QP channel

    #based on a linear fit after calculating the nQPs
    # using random sampling of the dispersion relation
    # given a Bose-Einstein distribution for nD/dp

    slope_t2, b_t2 = 1244.04521473, 29.10987933 #linear fit params
    return slope_t2*qp_energy + b_t2


def Get_V(filename):
    '''
    returns h which can be used to map momentum in eV to velocity in m/s
    '''
    Velocity_data = pd.read_table(filename, sep=",", usecols=['Momentum', 'velocity'])
    Velocity_data['Momentum']=Velocity_data['Momentum']*1e3
    Velocity_data['velocity']=Velocity_data['velocity']


    #h = interp1d(Velocity_data['Momentum'],Velocity_data['velocity'],fill_value=(0, 0), bounds_error=False)

    domega=np.polynomial.polynomial.Polynomial.fit(Velocity_data['Momentum'], Velocity_data['velocity'], 9, window=[0, 4900] )
    domega_params =domega.convert().coef


    #return h
    return domega_params


def dOmega(p,domega_params):
    '''
    returns h which can be used to map momentum in eV to velocity in m/s
    '''
    e=np.zeros(np.size(p))
    for i in range(len(domega_params)):
        e += domega_params[i]*pow(p, i)
    return e #returns energy in eV



def Boltzman_factor_without_V(p,T_in_K,omega_params):
    return 1.0/(np.exp(Omega(p,omega_params)/(T_in_K*0.025/300))-1)


def Energy_partion(TotalE_He,f,g,IR):
    singlet_fraction=np.zeros(np.size(TotalE_He))
    singlet_photon_mean=TotalE_He*f(TotalE_He)/0.016
    singlet_fraction =  np.random.poisson(singlet_photon_mean)*0.016/TotalE_He

    IR_fraction=np.zeros(np.size(TotalE_He))
    IR_photon_mean=TotalE_He*IR(TotalE_He)/(0.0001)
    IR_fraction =  np.random.poisson(IR_photon_mean)*(0.0001)/TotalE_He

    triplet_fraction=np.zeros(np.size(TotalE_He))
    triplet_photon_mean=TotalE_He*g(TotalE_He)/0.016
    triplet_fraction = np.random.poisson(triplet_photon_mean)*0.016/TotalE_He

    qp_fraction=np.zeros(np.size(TotalE_He))
    qp_fraction = 1.0-triplet_fraction-singlet_fraction-IR_fraction


    return singlet_fraction, singlet_fraction*TotalE_He/0.016 , IR_fraction,\
                           IR_fraction*TotalE_He/0.0001, qp_fraction, triplet_fraction*TotalE_He/0.016


def Get_MomentumDistribution(N,amplitude1, peak1,spread1, amplitude2,peak2, spread2):
    a=np.random.randint(2, size=N)
    b = np.zeros(np.size(a))
    b[a==0] = 1

    return np.random.normal(peak1, spread1, N)*a + np.random.normal(peak2, spread2, N)*b


#def Get_MomentumDistribution_UsingTemp(N,momentum_sampler):

    #return np.asarray(random.choices(p, functional_form, k=N))
#    return np.asarray(random.choices(momentum_sampler, k=N))

def Get_MomentumDistribution_UsingTemp(N,p,y):
    x=np.int_(y*N)
    x[-1]=x[-1]+ (N-np.sum(x))
    df = pd.DataFrame(({'Number': x[1:] ,\
          'momentum': p[1:]}) )

    df = df.loc[df.index.repeat(df['Number'])]

    return np.asarray(df['momentum'])

def Get_Energy_Velocity_Momentum_Position_df_from_recoil(Ein_QP_channel,x,y,z,p,pdf,omega_params,domega_params):
    '''
    We generate the intial QP particle velocity, momentum and energy information, we also return the intial position
    of the interaction.
    '''

    N= Get_NoOfQuasiparticles(Ein_QP_channel)


    Energy=np.zeros(int(N))
    Velocity=np.zeros(int(N))
    Momentum=np.zeros(int(N))

    #Momentum=Get_MomentumDistribution( int(2*N), 0.00116, 800, 100, 0.00116, 3800, 100)
    Momentum=Get_MomentumDistribution_UsingTemp(int(N), p,pdf)
    Energy=Omega(Momentum,omega_params)
    Velocity=dOmega(Momentum,domega_params)



    Energy=Energy[Momentum>1100]  # no finite lifetime QP
    Velocity=Velocity[Momentum>1100]
    Momentum=Momentum[Momentum>1100]

    Momentum=Momentum[Energy>0.00062] # QE threshold
    Velocity=Velocity[Energy>0.00062]
    Energy=Energy[Energy>0.00062]

    X=x*np.ones(np.size(Momentum))
    Y=y*np.ones(np.size(Momentum))
    Z=z*np.ones(np.size(Momentum))

    Intial_X=x
    Intial_Y=y
    Intial_Z=z






    return Energy,Momentum,Velocity,X,Y,Z,Intial_X,Intial_Y,Intial_Z

def Get_PromptEnergy_plot(N_Photons,X,Y,Z,Radius_Helium,Height_CPD):
    Cutoff_angle = 90
    unit_vector = np.random.uniform(low=-1, high=1,size=3*N_Photons).reshape(N_Photons, 3)
    nx=unit_vector.T[0]/np.linalg.norm(unit_vector, axis=1)
    ny=unit_vector.T[1]/np.linalg.norm(unit_vector, axis=1)
    nz=unit_vector.T[2]/np.linalg.norm(unit_vector, axis=1)
    status=np.full(np.size(nz), True)

    all_nx = 300*nx+X
    all_ny = 300*ny+Y
    all_nz = 300*nz+Z

    #plt.axes().set_aspect('equal')
    #plt.scatter(nz,nx)

    nx=nx[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]
    ny=ny[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]
    nz=nz[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]

    Z_He=Height_CPD*np.ones(np.size(nx))
    X_He=nx*(Height_CPD-Z)/nz + X
    Y_He=ny*(Height_CPD-Z)/nz + Y
    R_square=X_He**2+Y_He**2

    survived_singlet=X_He[R_square<Radius_Helium**2]



    return X_He[R_square<Radius_Helium**2] , Y_He[R_square<Radius_Helium**2], Z_He[R_square<Radius_Helium**2],\
                         all_nx, all_ny, all_nz



def Get_PromptEnergy_from_recoil_position(N_Photons,N_IR,X,Y,Z,Radius_Helium,Radius_CPD,Height_CPD):
    Cutoff_angle = 90
    u = np.random.uniform(low=0, high=1,size=int(N_Photons))
    theta=2*np.pi*u
    phi= np.arccos(1-2*u)

    nx=np.sin(phi)*np.cos(theta)
    ny=np.sin(phi)*np.sin(theta)
    nz=np.cos(phi)

    nx=nx[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]
    ny=ny[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]
    nz=nz[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]

    Z_He=(Height_CPD-6)*np.ones(np.size(nx))
    X_He=nx*(Height_CPD-6-Z)/nz + X
    Y_He=ny*(Height_CPD-6-Z)/nz + Y

    R_square=X_He**2+Y_He**2

    Z_He=Z_He[R_square<Radius_Helium**2]
    X_He=X_He[R_square<Radius_Helium**2]
    Y_He=Y_He[R_square<Radius_Helium**2]

    nx=nx[R_square<Radius_Helium**2]
    ny=ny[R_square<Radius_Helium**2]
    nz=nz[R_square<Radius_Helium**2]

    Z_He=(Height_CPD)*np.ones(np.size(Z_He))
    X_He=nx*(Height_CPD-Z)/nz + X
    Y_He=ny*(Height_CPD-Z)/nz + Y
    R_square=X_He**2+Y_He**2



    survived_singlet=X_He[R_square<Radius_CPD**2]





    u = np.random.uniform(low=0, high=1,size=int(N_IR))
    theta=2*np.pi*u
    phi= np.arccos(1-2*u)

    nx=np.sin(phi)*np.cos(theta)
    ny=np.sin(phi)*np.sin(theta)
    nz=np.cos(phi)

    nx=nx[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]
    ny=ny[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]
    nz=nz[ np.degrees( np.arccos(  nz*np.ones( np.size(nz) )  ) ) < Cutoff_angle]

    Z_He=(Height_CPD-6)*np.ones(np.size(nx))
    X_He=nx*(Height_CPD-6-Z)/nz + X
    Y_He=ny*(Height_CPD-6-Z)/nz + Y

    R_square=X_He**2+Y_He**2

    Z_He=Z_He[R_square<Radius_Helium**2]
    X_He=X_He[R_square<Radius_Helium**2]
    Y_He=Y_He[R_square<Radius_Helium**2]

    nx=nx[R_square<Radius_Helium**2]
    ny=ny[R_square<Radius_Helium**2]
    nz=nz[R_square<Radius_Helium**2]

    Z_He=(Height_CPD)*np.ones(np.size(Z_He))
    X_He=nx*(Height_CPD-Z)/nz + X
    Y_He=ny*(Height_CPD-Z)/nz + Y
    R_square=X_He**2+Y_He**2



    survived_IR=X_He[R_square<Radius_CPD**2]




    return 0.016*1000*np.size(survived_singlet)+ 0.0001*1000*np.size(survived_IR)


def Remove_the_downward_QP(Vx_unit_vector,Vy_unit_vector, Vz_unit_vector, X1, Y1, Z1,\
                          Energy,Momentum,Velocity):
    '''
    removes the QP's that are moving in negative direction based on cutoff angle, as there is no reflection QP
    is assumed here. As we decrease the size of numpy array, we need to pass all the the relevant np arrays.
    '''
    Cutoff_angle=89.5
    #select only particles with +ve Velocity and fill rest with zero and do the selection
    Vx_unit_vector_new_positive=np.where(Velocity>0,Vx_unit_vector,0)
    Vy_unit_vector_new_positive=np.where(Velocity>0,Vy_unit_vector,0)
    Vz_unit_vector_new_positive=np.where(Velocity>0,Vz_unit_vector,0)

    #select only particles with -ve Velocity, change their nx,ny,nz and fill rest with zero and do the selection
    Vx_unit_vector_new_negative=np.where(Velocity<0,-Vx_unit_vector,0)
    Vy_unit_vector_new_negative=np.where(Velocity<0,-Vy_unit_vector,0)
    Vz_unit_vector_new_negative=np.where(Velocity<0,-Vz_unit_vector,0)

    #add those two
    Vx_unit_vector=Vx_unit_vector_new_positive+Vx_unit_vector_new_negative
    Vy_unit_vector=Vy_unit_vector_new_positive+Vy_unit_vector_new_negative
    Vz_unit_vector=Vz_unit_vector_new_positive+Vz_unit_vector_new_negative


    X=X1[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    Y=Y1[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    Z=Z1[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    nx=Vx_unit_vector[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    ny=Vy_unit_vector[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    nz=Vz_unit_vector[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    Energy=Energy[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    Momentum=Momentum[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]
    Velocity=Velocity[ np.degrees( np.arccos(  Vz_unit_vector*np.ones( np.size(Vz_unit_vector) )  ) ) < Cutoff_angle]

    return X,Y,Z,nx, ny, nz,Energy,Momentum,Velocity


def get_Coordinate_He_Surface(X,Y,Z,nx, ny, nz,Energy,Momentum,Velocity, half_height_Helium,Radius_Helium):
    '''
    Based on 3-D geometry we get the point of interestion of a line(QP ray)
    and a plane He-surface plane(half_height_Helium). Once the intersection point is determined
    based on the Radius of Helium, we will see if that point is inside the CPD or not.
    '''

    Z_He=half_height_Helium*np.ones(np.size(Z))
    X_He=nx*(half_height_Helium-Z)/nz + X
    Y_He=ny*(half_height_Helium-Z)/nz + Y
    R_square=X_He**2+Y_He**2

    return  X_He[R_square<Radius_Helium**2],Y_He[R_square<Radius_Helium**2],Z_He[R_square<Radius_Helium**2]\
                  ,nx[R_square<Radius_Helium**2], ny[R_square<Radius_Helium**2], nz[R_square<Radius_Helium**2]\
                ,Energy[R_square<Radius_Helium**2],Momentum[R_square<Radius_Helium**2],Velocity[R_square<Radius_Helium**2]



def Get_Angle_of_incidence(nz_He):
    return  np.degrees(np.arccos(nz_He))




def Get_critical_Angle_of_incidence(Energy,Velocity):
    '''
    Calculates the critical angle for evaporation using snells law, and we replace the nan with zero crictical angle,
    no evaporation at such angles
    '''
    sin_critical_angle = np.sqrt( 2*(Energy-0.00062)/(4.002603254*1e9) )/(np.abs(Velocity)/3e8)
    critical_angle_deg=  np.degrees( np.arcsin(sin_critical_angle) )
    critical_angle_deg=np.nan_to_num(critical_angle_deg)
    return critical_angle_deg





def Remove_bad_incident_QP(X_He,Y_He,Z_He,nx_He, ny_He, nz_He,Energy,Momentum,Velocity,theta_I,critical_Angle_of_incidence):
    '''
    this is used to remove the QP whose angle of incident is greater critical angle of incidence
    '''

    X=X_He[ theta_I < critical_Angle_of_incidence]
    Y=Y_He[ theta_I < critical_Angle_of_incidence]
    Z=Z_He[ theta_I < critical_Angle_of_incidence]

    nx=nx_He[ theta_I < critical_Angle_of_incidence]
    ny=ny_He[ theta_I < critical_Angle_of_incidence]
    Energy=Energy[ theta_I < critical_Angle_of_incidence]
    Momentum=Momentum[ theta_I < critical_Angle_of_incidence]
    Velocity=Velocity[ theta_I < critical_Angle_of_incidence]
    nz=nz_He[ theta_I < critical_Angle_of_incidence]
    theta_I =theta_I[ theta_I < critical_Angle_of_incidence]

    return X,Y,Z,nx, ny, nz,Energy,Momentum,Velocity,theta_I



def convert_taniverse_cos(x):
    return np.where(x<0,180+x,x)

def change_direction(nx_He_S,ny_He_S,nz_He_S,X_He_S,Y_He_S,Z_He_S,theta_I_He_S, Energy,Velocity,Momentum):
    '''
    once we know the atom is allowed to quantumn evaporated, we need to find a new ray direction.
    Conserving momentum and energy.
    '''

    sin_theta_R = (Velocity/3e8)*np.sin(np.radians(theta_I_He_S))/np.sqrt( 2*(Energy-0.00062)/(4.002603254*1e9) )
    theta_R = np.arcsin(sin_theta_R)

    Velocity_He_atom=np.sqrt( 2*(Energy-0.00062)/(4.002603254*1e9) ) *3e8

    new_Vz=Velocity_He_atom*np.cos(theta_R)

    tan_degree=np.degrees(np.arctan(np.abs(nx_He_S/ny_He_S)))


    new_Vx= nx_He_S/np.abs(nx_He_S)*Velocity_He_atom*np.sin(theta_R)*np.cos((tan_degree))

    new_Vy= ny_He_S/np.abs(ny_He_S)*Velocity_He_atom*np.sin(theta_R)*np.sin((tan_degree))


    magnitude=np.sqrt(new_Vx**2+new_Vy**2+new_Vz**2)
    nx_V=new_Vx/magnitude
    ny_V=new_Vy/magnitude
    nz_V=new_Vz/magnitude

    # how much do you evaporate, 60%
    probability_Evaporation=np.random.binomial(1, 0.60, np.size(nx_He_S))

    return nx_V[probability_Evaporation>0],ny_V[probability_Evaporation>0],\
                   nz_V[probability_Evaporation>0], X_He_S[probability_Evaporation>0],\
                   Y_He_S[probability_Evaporation>0], Z_He_S[probability_Evaporation>0],\
        Energy[probability_Evaporation>0], Velocity[probability_Evaporation>0],Momentum[probability_Evaporation>0]



def get_Coordinate_CPD_Surface(X_He_S,Y_He_S,Z_He_S,nx_V,ny_V,nz_V,Energy,Momentum,Velocity, \
                              Height_CPD,Radius_Helium,Radius_CPD):
    '''
    Based on 3-D geometry we get the point of interestion of a line(evaporated ray)
    and a plane CPD-surface plane(half_height_CPD).
    '''

    #find X,Y,Z at where the copper hole ends
    Z_CPD=(Height_CPD-6)*np.ones(np.size(Z_He_S))
    X_CPD=nx_V*(Height_CPD-6-Z_He_S)/nz_V  + X_He_S
    Y_CPD=ny_V*(Height_CPD-6-Z_He_S)/nz_V  + Y_He_S

    Radius=X_CPD**2+Y_CPD**2

    Z_CPD=Z_CPD[Radius<Radius_Helium**2]
    X_CPD=X_CPD[Radius<Radius_Helium**2]
    Y_CPD=Y_CPD[Radius<Radius_Helium**2]
    X_He_S=X_He_S[Radius<Radius_Helium**2]
    Y_He_S=Y_He_S[Radius<Radius_Helium**2]
    Z_He_S=Z_He_S[Radius<Radius_Helium**2]
    Energy=Energy[Radius<Radius_Helium**2]
    Momentum=Momentum[Radius<Radius_Helium**2]
    Velocity=Velocity[Radius<Radius_Helium**2]
    nx_V=nx_V[Radius<Radius_Helium**2]
    ny_V=ny_V[Radius<Radius_Helium**2]
    nz_V=nz_V[Radius<Radius_Helium**2]


    #find X,Y,Z at the CPD
    Z_CPD=Height_CPD*np.ones(np.size(Z_He_S))
    X_CPD=nx_V*(Height_CPD-Z_He_S)/nz_V + X_He_S
    Y_CPD=ny_V*(Height_CPD-Z_He_S)/nz_V + Y_He_S
    Velocity_atom=np.sqrt(2*(Energy -0.00062)/(4.002603254*1e9))*3e8*1e3

    Radius=X_CPD**2+Y_CPD**2

    Z_CPD=Z_CPD[Radius<Radius_CPD**2]
    X_CPD=X_CPD[Radius<Radius_CPD**2]
    Y_CPD=Y_CPD[Radius<Radius_CPD**2]
    X_He_S=X_He_S[Radius<Radius_CPD**2]
    Y_He_S=Y_He_S[Radius<Radius_CPD**2]
    Z_He_S=Z_He_S[Radius<Radius_CPD**2]
    Energy=Energy[Radius<Radius_CPD**2]
    Momentum=Momentum[Radius<Radius_CPD**2]
    Velocity=Velocity[Radius<Radius_CPD**2]
    Velocity_atom=Velocity_atom[Radius<Radius_CPD**2]


    return  X_CPD,Y_CPD,Z_CPD,X_He_S,Y_He_S,Z_He_S\
                ,Energy,Momentum,Velocity,Velocity_atom



def get_Time_QP(X_He_S,Y_He_S,Z_He_S,X_CPD,Y_CPD,Z_CPD,Velocity,\
             Velocity_atom,Energy,Momentum,Intial_X,Intial_Y,Intial_Z):
    '''
    Velocity refers to the QP particle velocity.
    Velocity_atom refers to QE He atom velocity.
    Calculate the Time taken by the QP to reach the
    Once the intersection point is determined based on the Radius of Helium, we
    will see if that point is inside the CPD or not. And we will return the Energy(eV) and total_time(microseconds)
    '''

    time_helium =np.sqrt((Intial_X - X_He_S)**2+ (Intial_Y-Y_He_S)**2+(Intial_Z-Z_He_S)**2)/np.abs(Velocity)*(1/1e3)
    time_vacuum =np.sqrt((X_He_S - X_CPD)**2+ (Y_He_S - Y_CPD)**2+(Z_He_S - Z_CPD)**2)/np.abs(Velocity_atom)
    time_helium = time_helium /(1/1e6)
    time_vacuum  = time_vacuum  /(1/1e6)
    total_time=time_helium+time_vacuum
    return Energy,total_time,Intial_X,Intial_Y,Intial_Z

def get_timeStamp_AmplitudeMean_Sim(E):
    xdata =E[0]
    ydata = np.zeros(100)
    for i in range(100):
        ydata[i]= np.mean(E[1:][:].T[i])
    return xdata,ydata


def Get_Energy_Velocity_Momentum_Position_Dataframe_from_recoil(Ein_QP_channel,x,y,z,momentum_sampler,omega_params,domega_params):
    '''
    We generate the intial QP particle velocity, momentum and energy information, we also return the intial position
    of the interaction.
    '''

    N= Get_NoOfQuasiparticles(Ein_QP_channel)


    Energy=np.zeros(int(N))
    Velocity=np.zeros(int(N))
    Momentum=np.zeros(int(N))

    #Momentum=Get_MomentumDistribution( int(2*N), 0.00116, 800, 100, 0.00116, 3800, 100)
    Momentum=Get_MomentumDistribution_UsingTemp(int(N), momentum_sampler)
    Energy=Omega(Momentum,omega_params)
    Velocity=dOmega(Momentum,domega_params)


    #Total_energy= np.cumsum(Energy)

    #Energy = Energy[ ( Total_energy < Ein_QP_channel ) ] # energy should be less in evaporation channel
    #Momentum = Momentum[ ( Total_energy < Ein_QP_channel ) ]
    #Velocity = Velocity[ ( Total_energy < Ein_QP_channel ) ]



    Energy=Energy[Momentum>1100]  # no finite lifetime QP
    Velocity=Velocity[Momentum>1100]
    Momentum=Momentum[Momentum>1100]

    Momentum=Momentum[Energy>0.00062] # QE threshold
    Velocity=Velocity[Energy>0.00062]
    Energy=Energy[Energy>0.00062]
    X=x*np.ones(np.size(Momentum))
    Y=y*np.ones(np.size(Momentum))
    Z=z*np.ones(np.size(Momentum))

    Intial_X=x
    Intial_Y=y
    Intial_Z=z

    Live_Status=np.zeros(np.size(Momentum))

    #get random momentum vectors
    u = np.random.uniform(low=0, high=1,size=np.size(Momentum))
    theta=2*np.pi*u
    phi= np.arccos(1-2*u)

    Vx_unit_vector=np.sin(phi)*np.cos(theta)
    Vy_unit_vector=np.sin(phi)*np.sin(theta)
    Vz_unit_vector=np.cos(phi)


    df = pd.DataFrame(({'Energy1': Energy ,\
          'Momentum1': Momentum,\
          'Velocity1': Velocity,\
          'X0':Intial_X ,\
          'Y0': Intial_Y ,\
          'Z0': Intial_Z,\
          'nx':Vx_unit_vector,\
          'ny':Vy_unit_vector,\
          'nz':Vz_unit_vector,\
          'ReachHeSurface':Live_Status}) )

    df= df.rename(columns = {'index':'index'})



    return df

def get_interaction_point(df):
    r=30
    z=13.75
    a= df['nx']**2 + df['ny']**2
    b=  2*(df['nx'] +df['ny'])
    c= df['X0']**2+df['Y0']**2-r**2
    sol1 =( -b + np.sqrt(b**2 -4*a*c) )/ (2*a)
    sol2 =( -b - np.sqrt(b**2 -4*a*c) )/ (2*a)
    k=np.where(sol1>sol2, sol1,sol2) #because x0,y0 <R therefore products of root <0
    X1 = df['X0']+ k*df['nx']
    Y1 = df['Y0']+ k*df['ny']
    Z1 = df['Z0']+ k*df['nz']

    #X2=np.where( df['nz'] > 0.0, df['X0'] + df['nx'] * (z-df['Z0'])/(df['nz']), \
    #                                 df['X0'] + df['nx'] * (z-df['Z0'])/(df['nz'])  )

    Y2=np.where( df['nz'] < 0.0, df['Y0'] + df['ny'] * (-z-df['Z0'])/(df['nz']), \
                                             df['Y0'] + df['ny'] * (-z-df['Z0'])/(df['nz'])  )

    Z2=np.where( df['nz'] > 0.0, z, -z)

    distance1 = (X1- df['X0'])**2+ (Y1- df['Y0'])**2+(Z1- df['Z0'])**2

    distance2 = (X2- df['X0'])**2+ (Y2- df['Y0'])**2+(Z2- df['Z0'])**2

    newX=np.where( distance1 < distance2, X1 , X2)
    newY=np.where( distance1 < distance2, Y1 , Y2)
    newZ=np.where( distance1 < distance2, Z1 , Z2)

    return X1,Y1,Z1
    #return newX,newY,newZ





if __name__ == "__main__":

    print("Helper Functions")
