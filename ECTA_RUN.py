"""
Created on Tue Jan 28 16:15:16 2020

@author: hour
"""
import glob
import os, os.path
import numpy as np
np.set_printoptions(threshold=np.inf)
import time
from scipy.spatial import ConvexHull
import matplotlib
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
import warnings
#from scipy.io.matlab import mio
from sklearn.cluster import DBSCAN
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import netCDF4 as nc

start_time = time.time()


cmap_density = { 'red': (
               (0.000000,0.871094,0.871094),(0.015625,0.816406,0.816406),(0.031250,0.761719,0.761719),(0.046875,0.707031,0.707031),
               (0.062500,0.656250,0.656250),(0.078125,0.601562,0.601562),(0.093750,0.546875,0.546875),(0.109375,0.496094,0.496094),
               (0.125000,0.433594,0.433594),(0.140625,0.371094,0.371094),(0.156250,0.308594,0.308594),(0.171875,0.246094,0.246094),
               (0.187500,0.183594,0.183594),(0.203125,0.121094,0.121094),(0.218750,0.058594,0.058594),(0.234375,0.000000,0.000000),
               (0.250000,0.000000,0.000000),(0.265625,0.000000,0.000000),(0.281250,0.000000,0.000000),(0.296875,0.000000,0.000000),
               (0.312500,0.000000,0.000000),(0.328125,0.000000,0.000000),(0.343750,0.000000,0.000000),(0.359375,0.000000,0.000000),
               (0.375000,0.000000,0.000000),(0.390625,0.000000,0.000000),(0.406250,0.000000,0.000000),(0.421875,0.000000,0.000000),
               (0.437500,0.000000,0.000000),(0.453125,0.000000,0.000000),(0.468750,0.000000,0.000000),(0.484375,0.000000,0.000000),
               (0.500000,0.121094,0.121094),(0.515625,0.246094,0.246094),(0.531250,0.371094,0.371094),(0.546875,0.496094,0.496094),
               (0.562500,0.621094,0.621094),(0.578125,0.746094,0.746094),(0.593750,0.871094,0.871094),(0.609375,0.996094,0.996094),
               (0.625000,0.996094,0.996094),(0.640625,0.996094,0.996094),(0.656250,0.996094,0.996094),(0.671875,0.996094,0.996094),
               (0.687500,0.996094,0.996094),(0.703125,0.996094,0.996094),(0.718750,0.996094,0.996094),(0.734375,0.996094,0.996094),
               (0.750000,0.996094,0.996094),(0.765625,0.996094,0.996094),(0.781250,0.996094,0.996094),(0.796875,0.996094,0.996094),
               (0.812500,0.996094,0.996094),(0.828125,0.996094,0.996094),(0.843750,0.996094,0.996094),(0.859375,0.996094,0.996094),
               (0.875000,0.933594,0.933594),(0.890625,0.871094,0.871094),(0.906250,0.808594,0.808594),(0.921875,0.746094,0.746094),
               (0.937500,0.683594,0.683594),(0.953125,0.621094,0.621094),(0.968750,0.558594,0.558594),(1.000000,0.496094,0.496094),
               (1.000000,0.496094,0.496094)),
    'green': (
              (0.000000,0.871094,0.871094),(0.015625,0.816406,0.816406),(0.031250,0.761719,0.761719),(0.046875,0.707031,0.707031),
              (0.062500,0.656250,0.656250),(0.078125,0.601562,0.601562),(0.093750,0.546875,0.546875),(0.109375,0.496094,0.496094),
              (0.125000,0.433594,0.433594),(0.140625,0.371094,0.371094),(0.156250,0.308594,0.308594),(0.171875,0.246094,0.246094),
              (0.187500,0.183594,0.183594),(0.203125,0.121094,0.121094),(0.218750,0.058594,0.058594),(0.234375,0.000000,0.000000),
              (0.250000,0.074219,0.074219),(0.265625,0.152344,0.152344),(0.281250,0.230469,0.230469),(0.296875,0.308594,0.308594),
              (0.312500,0.386719,0.386719),(0.328125,0.464844,0.464844),(0.343750,0.542969,0.542969),(0.359375,0.621094,0.621094),
              (0.375000,0.667969,0.667969),(0.390625,0.714844,0.714844),(0.406250,0.761719,0.761719),(0.421875,0.808594,0.808594),
              (0.437500,0.855469,0.855469),(0.453125,0.902344,0.902344),(0.468750,0.949219,0.949219),(0.484375,0.996094,0.996094),
              (0.500000,0.996094,0.996094),(0.515625,0.996094,0.996094),(0.531250,0.996094,0.996094),(0.546875,0.996094,0.996094),
              (0.562500,0.996094,0.996094),(0.578125,0.996094,0.996094),(0.593750,0.996094,0.996094),(0.609375,0.996094,0.996094),
              (0.625000,0.933594,0.933594),(0.640625,0.871094,0.871094),(0.656250,0.808594,0.808594),(0.671875,0.746094,0.746094),
              (0.687500,0.683594,0.683594),(0.703125,0.621094,0.621094),(0.718750,0.558594,0.558594),(0.734375,0.496094,0.496094),
              (0.750000,0.433594,0.433594),(0.765625,0.371094,0.371094),(0.781250,0.308594,0.308594),(0.796875,0.246094,0.246094),
              (0.812500,0.183594,0.183594),(0.828125,0.121094,0.121094),(0.843750,0.058594,0.058594),(0.859375,0.000000,0.000000),
              (0.875000,0.000000,0.000000),(0.890625,0.000000,0.000000),(0.906250,0.000000,0.000000),(0.921875,0.000000,0.000000),
              (0.937500,0.000000,0.000000),(0.953125,0.000000,0.000000),(0.968750,0.000000,0.000000),(1.000000,0.000000,0.000000),
              (1.000000,0.000000,0.000000)),
        'blue': (
             (0.000000,0.871094,0.871094),(0.015625,0.886719,0.886719),(0.031250,0.906250,0.906250),(0.046875,0.921875,0.921875),
             (0.062500,0.941406,0.941406),(0.078125,0.957031,0.957031),(0.093750,0.976562,0.976562),(0.109375,0.996094,0.996094),
             (0.125000,0.996094,0.996094),(0.140625,0.996094,0.996094),(0.156250,0.996094,0.996094),(0.171875,0.996094,0.996094),
             (0.187500,0.996094,0.996094),(0.203125,0.996094,0.996094),(0.218750,0.996094,0.996094),(0.234375,0.996094,0.996094),
             (0.250000,0.949219,0.949219),(0.265625,0.902344,0.902344),(0.281250,0.855469,0.855469),(0.296875,0.808594,0.808594),
             (0.312500,0.761719,0.761719),(0.328125,0.714844,0.714844),(0.343750,0.667969,0.667969),(0.359375,0.621094,0.621094),
             (0.375000,0.542969,0.542969),(0.390625,0.464844,0.464844),(0.406250,0.386719,0.386719),(0.421875,0.308594,0.308594),
             (0.437500,0.230469,0.230469),(0.453125,0.152344,0.152344),(0.468750,0.074219,0.074219),(0.484375,0.000000,0.000000),
             (0.500000,0.000000,0.000000),(0.515625,0.000000,0.000000),(0.531250,0.000000,0.000000),(0.546875,0.000000,0.000000),
             (0.562500,0.000000,0.000000),(0.578125,0.000000,0.000000),(0.593750,0.000000,0.000000),(0.609375,0.000000,0.000000),
             (0.625000,0.000000,0.000000),(0.640625,0.000000,0.000000),(0.656250,0.000000,0.000000),(0.671875,0.000000,0.000000),
             (0.687500,0.000000,0.000000),(0.703125,0.000000,0.000000),(0.718750,0.000000,0.000000),(0.734375,0.000000,0.000000),
             (0.750000,0.000000,0.000000),(0.765625,0.000000,0.000000),(0.781250,0.000000,0.000000),(0.796875,0.000000,0.000000),
             (0.812500,0.000000,0.000000),(0.828125,0.000000,0.000000),(0.843750,0.000000,0.000000),(0.859375,0.000000,0.000000),
             (0.875000,0.058594,0.058594),(0.890625,0.121094,0.121094),(0.906250,0.183594,0.183594),(0.921875,0.246094,0.246094),
             (0.937500,0.308594,0.308594),(0.953125,0.371094,0.371094),(0.968750,0.433594,0.433594),(1.000000,0.496094,0.496094),
             (1.000000,0.496094,0.496094))}


cmap_density = LinearSegmentedColormap('my_colormap', cmap_density, 100)
cmap_jet=plt.get_cmap('jet')

def Convert_time(minutes):  #Minutes since beginning of the day in hh:mm UTC
    hour, min = divmod(minutes, 60) 
    return "%d:%02d" % (hour, min) 

params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 15, # fontsize for x and y labels (was 10)
    'axes.titlesize': 20,
    'font.size': 15, # was 10
    'legend.fontsize': 15, # was 10
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': False,
    'figure.figsize': [19.2, 10.8],    
    'font.family': 'sans-serif',                
    #Tight layout pour optimiser l'espace entre les subplots 
    
    #Figsize * dpi= taille figure en pixels (sans tight,bbox), si besoin de précision et de zoomer prendre grosse résolution
    
    #BBox inches=coupe le blanc autour de la figure, réduit la resolution en pixels
}
matplotlib.rcParams.update(params)

###############################################################
                        #File reading
###############################################################

warnings.filterwarnings("ignore") 

#DENSITIES computation parameters
Delta_x=1000 #m           #Defaut 1000
Delta_y=1000 #m           #Defaut 1000
Delta_z=0.05 #Km          #Defaut 0.05 
Time_interval=5 #min
THLD=25 # limit fixed to 25% of the maximum FED of the cell for cell borders
One_pixel_area=Delta_x*Delta_y
TEST=1

def ECTA(DAY,domain,T_start,T_end,LMA,Wdir):    # multiple return when no data     
    # DOMAIN AND TRACKING PERIOD 
    if LMA=='SAETTA':
        FilesName='L2b.V02.EXAEDRE.SAETTA.MTRG.'
        GRID_LATLON2D_CORSICA_1km=np.load(Path(Wdir/'GRID_LATLON2D_CORSICA_1km.npz'))  #GRID over the Corsican domain (1km*2) 
        LON2D=GRID_LATLON2D_CORSICA_1km['LON2D']  #to get that, i did meshgrid with XD and YD --> 2D array in meter and then m(X2D,Y2D)
        LAT2D=GRID_LATLON2D_CORSICA_1km['LAT2D']
        #COORDINATES associated with the 2D grid 
        lat_min=41     
        lat_max=43.5
        lon_min=6.5         
        lon_max=10.6

    if LMA=='HYLMA':
        FilesName='L2b.V02.Hy_SOP1.Hy_LMA.MTRG.'
        GRID_LATLON2D_SOUTHFRANCE_1km=np.load(Path(Wdir/'GRID_LATLON2D_SOUTHFRANCE_1km.npz'))  #GRID over the HYLMA domain (1km*2) 
        LON2D=GRID_LATLON2D_SOUTHFRANCE_1km['LON2D']  #to get that, i did meshgrid with XD and YD --> 2D array in meter and then m(X2D,Y2D)
        LAT2D=GRID_LATLON2D_SOUTHFRANCE_1km['LAT2D']
        #COORDINATES associated with the 2D grid 
        lat_min=42.6 
        lat_max=45.3
        lon_min=2.5       
        lon_max=6    

    
    print('START ECTA3D: CELL IDENTIFICATION AND TRACKING')
    print(DAY)
    try:
        # Create target Directory
        os.mkdir(Path(Wdir/DAY/'ECTA'))
        print("Directory ECTA Created ") 
    except FileExistsError:
        print("Directory ECTA already exists")
    
    Continuous_activity=0 #if ECTA applied on multiples days, no follow up between 2 days
    if len(sorted(glob.glob(str(Path(Wdir/DAY/'L2b_*')))))==0:
        print('NO DATA FOR THIS DAY \n')
        np.save(Path(Wdir/DAY/'ECTA'/'ECTA_CELL_data_'+domain+'_10km.npy'), -999)
        return Continuous_activity  #def
       
    
    ########################## PREPARATION of the list of L2b files to open ###########################################
    date_DAY=datetime(2000+int(DAY[0:2]),int(DAY[2:4]),int(DAY[4:6]))
    Year=str(date_DAY.year)
    Month=str("%02d" % date_DAY.month)
    Day=str("%02d" % date_DAY.day)
    
    t_min=int(T_start/60)
    t_max=int(T_end/60)
    
    Paths=[]
    Big_LMA_time=np.empty(0) 
    Big_LMA_time_period=np.empty(0)   
    Big_LMA_lat_tmp=np.empty(0) 
    Big_LMA_lon_tmp=np.empty(0) 
    Big_LMA_nb_station=np.empty(0) 
    Big_LMA_chi2=np.empty(0)
    Big_LMA_flash=np.empty(0) 
    Big_LMA_alt=np.empty(0)
    Half_Diag=np.sqrt(Delta_x**2+Delta_y**2)/2 #To buffer the polygon created
    c=0
    A_nb_flash=0
    
    for i in np.arange(t_min, t_max+1):
        if i<10:
            Paths.append(sorted(glob.glob(str(Path(Wdir,DAY,FilesName+Year+Month+Day+'_0'+str(i)+'*')))))
        if i>=10:
            Paths.append(sorted(glob.glob(str(Path(Wdir,DAY,FilesName+Year+Month+Day+'_'+str(i)+'*')))))
    
    Paths_day = [item for sublist in Paths for item in sublist]
    #test if there is activity around 00UTC              
    if Paths_day[-1].startswith(str(Path(Wdir,DAY,FilesName+Year+Month+Day+'_2350'))):
            print('Evening late  activity, loading next day files')
            date_next_DAY=date_DAY+ timedelta(days=1)
            N_Date_full_formated=str(date_next_DAY.year)+str("%02d" % date_next_DAY.month)+str("%02d" % date_next_DAY.day)
            if os.path.exists(Path(Wdir/N_Date_full_formated)):
                #N for next day 
                Continuous_activity=1 #if activity continues on 2 days, follow tracking the next day until 3 am 
                N_Year=str(date_next_DAY.year)
                N_Month=str("%02d" % date_next_DAY.month)
                N_Day=str("%02d" % date_next_DAY.day)
                for i in np.arange(0,3): #Arbitrary at this moment, we add 3 hour of actvity for the next day (2h59 the limit) 
                    Paths.append(sorted(glob.glob(str(Path(Wdir,N_Date_full_formated,FilesName+N_Year+N_Month+N_Day+'_0'+str(i)+'*')))))
            else:
                print('NO FILES FOR THE NEXT DAY')   
                   
    ######################################## L2b Files loading #######################################################
    PATHS = [item for sublist in Paths for item in sublist]
    c1=0
    for files_path in PATHS:  #windows
        print(files_path)
        #L2b_file= mio.loadmat(files_path)
        ds_L2b = nc.Dataset(files_path)
        #print(ds_L2b)
        LMA_data=ds_L2b['lma_data'][:,:].data
        #print(LMA_data)
        #LMA_data=L2b_file['lma_data']
        if LMA_data.size==0:
            continue
        LMA_data[0,:]=LMA_data[0,:]-366 #Correct time
        LMA_time=LMA_data[0,:]
        if c1==0:
            START_DAY=np.intc(LMA_time)[0]
            c1=c1+1
        LMA_time_period=(LMA_time-START_DAY)*24.*60.  #Minute depuis le début de la journée
        LMA_lat_tmp=LMA_data[1,:]
        LMA_lon_tmp=LMA_data[2,:]
        LMA_alt=LMA_data[3,:]
        LMA_chi2=LMA_data[4,:]       
        LMA_nb_station=LMA_data[7,:]
        LMA_flash=LMA_data[8,:].astype(np.int64)
        
        #BIG files accumulate data from the multiples 10 min-files 
        Big_LMA_time=np.append(Big_LMA_time,LMA_time)
        Big_LMA_time_period=np.append(Big_LMA_time_period,LMA_time_period)
        Big_LMA_alt=np.append(Big_LMA_alt,LMA_alt)
        Big_LMA_lat_tmp=np.append(Big_LMA_lat_tmp,LMA_lat_tmp)
        Big_LMA_lon_tmp=np.append(Big_LMA_lon_tmp,LMA_lon_tmp)
        Big_LMA_nb_station=np.append(Big_LMA_nb_station,LMA_nb_station)
        Big_LMA_chi2=np.append(Big_LMA_chi2,LMA_chi2)
        c=np.max(LMA_flash)+1
        if A_nb_flash==0:
            N=np.max(LMA_flash)+1   #Number max of flash in this period 
        N=N+c
        for f in (np.unique(LMA_flash)): #Total Flash + max period to avoid problems    
            LMA_flash[LMA_flash==f]=N 
            N=N+1
        A_nb=len(np.unique(LMA_flash))
        A_nb_flash=A_nb_flash+A_nb      
        Big_LMA_flash=np.append(Big_LMA_flash,LMA_flash)
        N=np.max(Big_LMA_flash)+1 
   
    
   ####################################### FILTER on data ###############################################################
    #LMA
    #DOMAIN FILTER 
    Big_LMA_lat=Big_LMA_lat_tmp[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_lon=Big_LMA_lon_tmp[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_alt=Big_LMA_alt[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_time=Big_LMA_time[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_time_period=Big_LMA_time_period[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_nb_station=Big_LMA_nb_station[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_chi2=Big_LMA_chi2[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    Big_LMA_flash=Big_LMA_flash[np.logical_and(np.logical_and(Big_LMA_lon_tmp>lon_min, Big_LMA_lon_tmp<lon_max),np.logical_and(Big_LMA_lat_tmp>lat_min,Big_LMA_lat_tmp<lat_max))]
    print('Lat lon filter',len(np.unique(Big_LMA_flash)))
    
    #Sources dected by at least 7 stations
    Big_LMA_time=Big_LMA_time[Big_LMA_nb_station>=7]
    Big_LMA_time_period=Big_LMA_time_period[Big_LMA_nb_station>=7]
    Big_LMA_alt=Big_LMA_alt[Big_LMA_nb_station>=7]
    Big_LMA_lat=Big_LMA_lat[Big_LMA_nb_station>=7]
    Big_LMA_lon=Big_LMA_lon[Big_LMA_nb_station>=7]
    Big_LMA_chi2=Big_LMA_chi2[Big_LMA_nb_station>=7]
    Big_LMA_flash=Big_LMA_flash[Big_LMA_nb_station>=7]
    print('stations filter',len(np.unique(Big_LMA_flash)))
    
    #chi2>0.5
    Big_LMA_time=Big_LMA_time[Big_LMA_chi2<0.5]
    Big_LMA_time_period=Big_LMA_time_period[Big_LMA_chi2<0.5]
    Big_LMA_alt=Big_LMA_alt[Big_LMA_chi2<0.5]
    Big_LMA_lat=Big_LMA_lat[Big_LMA_chi2<0.5]
    Big_LMA_lon=Big_LMA_lon[Big_LMA_chi2<0.5]
    Big_LMA_flash=Big_LMA_flash[Big_LMA_chi2<0.5]
    print('chi2 filter',len(np.unique(Big_LMA_flash)))
    
    #Flashes with less than 10 sources removed
    TH_sources=10
    unique, counts = np.unique(Big_LMA_flash, return_counts=True)  
    Long_flashes=unique[counts>=TH_sources]    
    Big_LMA_long_flash_condition=np.isin(Big_LMA_flash,Long_flashes) 
    Big_LMA_time=Big_LMA_time[Big_LMA_long_flash_condition]
    Big_LMA_alt=Big_LMA_alt[Big_LMA_long_flash_condition]
    Big_LMA_time_period=Big_LMA_time_period[Big_LMA_long_flash_condition]
    Big_LMA_lat=Big_LMA_lat[Big_LMA_long_flash_condition]
    Big_LMA_lon=Big_LMA_lon[Big_LMA_long_flash_condition]
    Big_LMA_flash=Big_LMA_flash[Big_LMA_long_flash_condition]
    print('small flashes  filter',len(np.unique(Big_LMA_flash)))
   
    if len(Big_LMA_flash)==0: #no data in the domain for the day 
         np.save(Path(Wdir,DAY,'ECTA','ECTA_CELL_data_'+domain+'_10km.npy'), -999)
         print('NO CELLS \n')
         return Continuous_activity  #def
    
    #################################### 2D FED CALCULATION - Each minute ################################
    m = Basemap(projection='merc',resolution='h',fix_aspect=True,llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max)
    DeltaT=np.arange(int(np.min(Big_LMA_time_period)),int(np.max(Big_LMA_time_period))+2,1)  #Units: MINUTE SINCE BEGINNING OF THE DAY 
    xmin_LMA,ymin_LMA=m(lon_min,lat_min)
    xmax_LMA,ymax_LMA=m(lon_max,lat_max) 
    X_D=np.arange(xmin_LMA,xmax_LMA-Delta_x,Delta_x)
    Y_D=np.arange(ymin_LMA,ymax_LMA-Delta_y,Delta_y)
    X_D_T=np.arange(xmin_LMA,xmax_LMA,Delta_x)  #just for the histogram (1 value added)
    Y_D_T=np.arange(ymin_LMA,ymax_LMA,Delta_y)
    FED_TIME=np.zeros((len(X_D),len(Y_D),len(DeltaT)))
    
    i0=0
    print('2D FED COMPUTATION')
    for i in DeltaT:
        print(Convert_time(i))
        #Selectioning data in the Time interval 
        LAT=Big_LMA_lat[(Big_LMA_time_period<=i) & (Big_LMA_time_period>i-Time_interval)]  ##### AS RADAR, SO T is for obervation between T-5 min and T 
        LON=Big_LMA_lon[(Big_LMA_time_period<=i) & (Big_LMA_time_period>i-Time_interval)]  
        x_LMA,y_LMA=m(LON,LAT) #tranform lat lon in meter in the basemap projection 
        Flash=Big_LMA_flash[(Big_LMA_time_period<=i) & (Big_LMA_time_period>i-Time_interval)]
        FED=np.zeros((len(X_D),len(Y_D)))  
        for f in np.unique(Flash):   
            LMA_density_flash,x_bins,y_bins=np.histogram2d(x_LMA[Flash==f],y_LMA[Flash==f],[X_D_T,Y_D_T])
            LMA_density_flash=np.where(LMA_density_flash>0,1,0)
            FED=FED+LMA_density_flash
        FED_TIME[:,:,i0]=FED
        i0=i0+1
    
    del Big_LMA_lat,Big_LMA_lon, Big_LMA_alt,Big_LMA_flash  #free some RAM 
    print("FED on grid computed - START TRACKING")
    
    #########################################  START OF THE TRACKING ALGORITHM #########################################################
    THLD=25
    Delta_D=10000
    Run_time_list=[]
    Cells_Duration_list=[]
    Cells_speed_list=[]
    print('Adaptative THLD:',THLD,'%')
    print(str(int(Delta_D/1000))+' km DeltaD')
    run_start=time.time()
    flash_max=0
    Big_time=np.empty(0)
    Big_clust=np.empty(0)
    Big_x=np.empty(0)
    Big_y=np.empty(0)
    Centroid_x=np.empty(0)
    Centroid_y=np.empty(0)
    Area_pixels=np.empty(0)
    Cell_area=np.empty(0)
    Big_Nb_flash_clust=np.empty(0)
    Big_Max_flash_X=np.empty(0)
    Big_Max_flash_Y=np.empty(0)
    i1=0
    for i in DeltaT:  #loop on minutes 
        print(Convert_time(i))
        
        Pixel_loc_x,Pixel_loc_y=np.where(FED_TIME[:,:,i1]>0)  #pixels with FED 
        Pixel_NB_Flash=FED_TIME[FED_TIME[:,:,i1]>0][:,i1]   
        i1=i1+1
        
    
        if len(Pixel_NB_Flash)>0:
            if flash_max<np.max(Pixel_NB_Flash):
                flash_max=np.max(Pixel_NB_Flash)
        Pixel_loc_x=Pixel_loc_x*Delta_x+Delta_x/2
        Pixel_loc_y=Pixel_loc_y*Delta_y+Delta_y/2
        Pixel_loc=np.zeros((len(Pixel_loc_x),2))
        Pixel_loc[:,0]=Pixel_loc_x
        Pixel_loc[:,1]=Pixel_loc_y
        Adapt_tresh=np.zeros(len(Pixel_NB_Flash))
    
        
        # FIRST DBSCAN ON PIXELS 
        if Pixel_loc_x.size>1:   
            clustering1 = DBSCAN(eps=Delta_D, min_samples=1).fit(Pixel_loc)
            Cluster1=clustering1.labels_
    
            
            #Adaptative treshold - 2nd DBSCAN on previous clusters
            for id_cluster1 in np.unique(Cluster1): #loop on differents clusters
                Adapt_tresh[Cluster1==id_cluster1]=(THLD/100)*np.max(Pixel_NB_Flash[Cluster1==id_cluster1]) 
                
            Pixel_loc_x=Pixel_loc_x[Pixel_NB_Flash>Adapt_tresh]
            Pixel_loc_y=Pixel_loc_y[Pixel_NB_Flash>Adapt_tresh]
            Pixel_loc=np.zeros((len(Pixel_loc_x),2))
            Pixel_loc[:,0]=Pixel_loc_x
            Pixel_loc[:,1]=Pixel_loc_y
            clustering2 = DBSCAN(eps=Delta_D, min_samples=1).fit(Pixel_loc)
            Cluster=clustering2.labels_
            Cluster_NB_flash=Pixel_NB_Flash[Pixel_NB_Flash>Adapt_tresh]
    
            for id_cluster in np.unique(Cluster): #loop on differents clusters
                XX=Pixel_loc_x[Cluster==id_cluster]
                YY=Pixel_loc_y[Cluster==id_cluster]
                Cluster_2D=np.zeros((len(XX),2))
                Cluster_2D[:,0]=XX
                Cluster_2D[:,1]=YY
                Nb_flash_max_per_cluster=np.max(Cluster_NB_flash[Cluster==id_cluster])
                Max_flash_X=np.mean(XX[Cluster_NB_flash[Cluster==id_cluster]==Nb_flash_max_per_cluster]) #test at this moment, get a position of the center of pixels(s) with nb flash max
                Max_flash_Y=np.mean(YY[Cluster_NB_flash[Cluster==id_cluster]==Nb_flash_max_per_cluster])
                
                if len(Cluster[Cluster==id_cluster])>2: #clusters with at least 2 pixels, Polygon function bug with only 2 pixels
                    try:
                        Conv_surf=ConvexHull(Cluster_2D) # Create a convex surface with all pixels inside 
                        Contour=np.zeros((len(XX[Conv_surf.vertices]),2))   #get the vertices of this surface 
                        Contour[:,0]=XX[Conv_surf.vertices]
                        Contour[:,1]=YY[Conv_surf.vertices]
                        Poly=Polygon(list(tuple(Contour)))    #Polygon with the center of pixels
                        Poly_bigger=Poly.buffer(Half_Diag)   #Polygon with the entire pixels (half diag added)
                        Borders_area=Poly_bigger.area
                        Borders=np.asarray(Poly_bigger.exterior.xy)  #2D array with the coordinates of the borders of the second bigger polygon          
                        Centroid=np.asarray(Poly_bigger.centroid)
                        
                    except:  #If pixels are colinear (lign, diagonal), polygon function can t make a polygon of it 
                        Centroid=np.zeros(2)
                        Borders=np.zeros((2,5))   #5 points to make closed borders, rectangular (or square) border, 5 points to make it complete 
                        Borders[0,:]=[np.min(XX)-Delta_x/2,np.max(XX)+Delta_x/2,np.max(XX)+Delta_x/2,np.min(XX)-Delta_x/2,np.min(XX)-Delta_x/2]
                        Borders[1,:]=[np.min(YY)-Delta_y/2,np.min(YY)-Delta_y/2,np.max(YY)+Delta_y/2,np.max(YY)+Delta_y/2,np.min(YY)-Delta_y/2]                  
                        Centroid=[np.min(XX)+(np.max(XX)-np.min(XX))/2,np.min(YY)+(np.max(YY)-np.min(YY))/2]
                        Borders_area=(np.max(Borders[1,:])-np.min(Borders[1,:]))*(np.max(Borders[0,:])-np.min(Borders[0,:]))
                        #A='Linear shape '+Convert_time(i)+'-Cluster '+str(id_cluster)
                        #print(A)
                
                if len(Cluster[Cluster==id_cluster])<=2:    #clusters with one or two pixels
                    Centroid=np.zeros(2)
                    Borders=np.zeros((2,5))   #5 points to make closed borders, rectangular (or square) border, 5 points to make it complete 
                    Borders[0,:]=[np.min(XX)-Delta_x/2,np.max(XX)+Delta_x/2,np.max(XX)+Delta_x/2,np.min(XX)-Delta_x/2,np.min(XX)-Delta_x/2]
                    Borders[1,:]=[np.min(YY)-Delta_y/2,np.min(YY)-Delta_y/2,np.max(YY)+Delta_y/2,np.max(YY)+Delta_y/2,np.min(YY)-Delta_y/2]                  
                    Centroid=[np.min(XX)+(np.max(XX)-np.min(XX))/2,np.min(YY)+(np.max(YY)-np.min(YY))/2]
                    Borders_area=(np.max(Borders[1,:])-np.min(Borders[1,:]))*(np.max(Borders[0,:])-np.min(Borders[0,:]))
               
                
                Time=np.zeros(len(Borders[0,:]))
                clust=np.zeros(len(Borders[0,:]))
                x=np.zeros(len(Borders[0,:]))
                y=np.zeros(len(Borders[0,:]))
                Nb_flash_clust=np.zeros(len(Borders[0,:]))
            
                for j in range(len(Borders[0,:])):
                    Time[j]=i
                    clust[j]=int(id_cluster)
                    x[j]=Borders[0,j]
                    y[j]=Borders[1,j]
                    Nb_flash_clust[j]=int(Nb_flash_max_per_cluster)
                    
                Big_time=np.append(Big_time,Time)
                Big_clust=np.append(Big_clust,clust)
                Big_x=np.append(Big_x,x)
                Big_y=np.append(Big_y,y)
                Centroid_x=np.append(Centroid_x,np.repeat(Centroid[0],len(Borders[0,:])))
                Centroid_y=np.append(Centroid_y,np.repeat(Centroid[1],len(Borders[0,:])))
                Area_pixels=np.append(Area_pixels,np.repeat(XX.size*One_pixel_area/10**6,len(Borders[0,:])))  #in km2
                Cell_area=np.append(Cell_area,np.repeat(Borders_area/10**6,len(Borders[0,:])))
                Big_Max_flash_X=np.append(Big_Max_flash_X,np.repeat(Max_flash_X,len(Borders[0,:])))
                Big_Max_flash_Y=np.append(Big_Max_flash_Y,np.repeat(Max_flash_Y,len(Borders[0,:])))
                Big_Nb_flash_clust=np.append(Big_Nb_flash_clust,Nb_flash_clust)
            
           
    ## CELLS DATA for the period in one array (12 parameters * the number of vertices in the polygons for each cells)
    Cells=np.zeros((len(Big_x),12)) #correspond to each vertices of the polygons 
    Cells[:,0]=Big_time #time (minute since beginning of the day) (same value for each vertice of a cluster)
    Cells[:,1]=Big_clust #cluster id  (same value for each vertice of a cluster)
    Cells[:,2]=Big_x #X position of the vertices according to the basemap projection 
    Cells[:,3]=Big_y #Y position of the verices according to the basemap projection 
    Cells[:,4]=Centroid_x #Y position of the cell centroid according to the basemap projection (same value for each vertice of a cluster)
    Cells[:,5]=Centroid_y #Y position of the cell centroid according to the basemap projection (same value for each vertice of a cluster) 
    Cells[:,6]=Area_pixels #area defined by the some of the pixels  (same value for each vertice of a cluster)
    Cells[:,7]=Cell_area  #area defined by the area inside polygons  (same value for each vertice of a cluster)
    Cells[:,8]=Area_pixels/Cell_area #ratio of areas
    Cells[:,9]=Big_Max_flash_X # X position of the pixel with the max of flashes (same value for each vertice of a cluster)
    Cells[:,10]=Big_Max_flash_Y # Y position of the pixel with the max of flashes (same value for each vertice of a cluster)
    Cells[:,11]=Big_Nb_flash_clust # Number of maximum flashes in a pixel (same value for each vertice of a cluster)
    
   
    print('Geographic overlaping of clusters with time  --- %s seconds ---' % (time.time() - start_time))
       
    Period=np.unique(Cells[:,0]).astype(int) #TIME WITH CLUSTERS OBTAINED 
    
    ############################################  GEOGRAPHIC OVERLAPING CLUSTERS with TIME ##################################        
    #Clusters id will be changed in accordance with overlaping  
    TIME=np.intc(Big_LMA_time[0])+Period/(24.*60.)
    T=np.copy(Cells[:,0])
    Cl=np.copy(Cells[:,1])
    Cl_clean=np.copy(Cl) ## WILL BE NEW CLUSTERS ID
    prev_max=np.max(np.unique(Cells[Cells[:,0]==Period[0]][:,1])).astype(int)
    for id_period in range(1,len(Period)): #loop on time (minutes)
        print('T (minutes since start of the Day):',Period[id_period])
        ID_CLUST_PREV=np.unique(Cells[(Cells[:,0]<=Period[id_period]-1) & (Cells[:,0]>Period[id_period]-15)][:,1]) # take the cells ID in the 20 previous min (15 min + 5 min of integrated time window )
        PERIOD_CLUST_PREV=np.zeros(len(ID_CLUST_PREV))
        for I2 in range(len(ID_CLUST_PREV)):
            PERIOD_CLUST_PREV[I2]=np.max(Cells[:,0][(Cells[:,1]==ID_CLUST_PREV[I2]) & (Cells[:,0]<=Period[id_period]-1)])
        Prev=np.copy(ID_CLUST_PREV)   
        Actual=np.unique(Cells[Cells[:,0]==Period[id_period]][:,1]).astype(int)
        Matrice=np.zeros((len(Prev),len(Actual)))
        Mat_area=np.zeros((len(Prev),len(Actual)))
        if len(Actual)==0:
            continue
        if len(Prev)==0:
            New_Actual=Actual+prev_max+1
            for i in range(len(Actual)):
                Cl_clean[(T==Period[id_period])&(Cl==Actual[i])]=New_Actual[i]
            Cells[:,1]=Cl_clean
            continue
        
        for id_clust_prev in ID_CLUST_PREV:
            X_prev=Cells[(Cells[:,0]==PERIOD_CLUST_PREV[ID_CLUST_PREV==id_clust_prev]) & (Cells[:,1]==id_clust_prev)][:,2]    #first period, we don t touch to the clusters numbers
            Y_prev=Cells[(Cells[:,0]==PERIOD_CLUST_PREV[ID_CLUST_PREV==id_clust_prev]) & (Cells[:,1]==id_clust_prev)][:,3]
            XY_prev=list(tuple(np.array((X_prev,Y_prev)).T))
            poly_prev=Polygon(XY_prev)
            
            for id_clust in np.unique(Cells[Cells[:,0]==Period[id_period]][:,1]):
                X=Cells[np.logical_and(Cells[:,0]==Period[id_period],Cells[:,1]==id_clust)][:,2]    
                Y=Cells[np.logical_and(Cells[:,0]==Period[id_period],Cells[:,1]==id_clust)][:,3] 
                XY=list(tuple(np.array((X,Y)).T))
                poly=Polygon(XY)
                Test=poly_prev.overlaps(poly)
                if Test==1:  
                    Mat_area[Prev==id_clust_prev,Actual==id_clust]=poly_prev.intersection(poly).area
                    
                if Test==0:
                    if poly_prev.contains(poly)==1:
                        Mat_area[Prev==id_clust_prev,Actual==id_clust]=poly_prev.intersection(poly).area
                    
                    elif poly.contains(poly_prev)==1:  
                        Mat_area[Prev==id_clust_prev,Actual==id_clust]=poly_prev.intersection(poly).area
                    
                    else:    
                          Matrice[Prev==id_clust_prev,Actual==id_clust]=0
        New_Actual=np.zeros(len(Actual),dtype=int)    
        
        #loop on ligns and columns of the matrice to make only one 1 per lign/column         
        #ligns
        for id_l_Matrice in range(len(Mat_area[:,0])):
            Max_area=np.max(Mat_area[id_l_Matrice,:])
            Mat_area[id_l_Matrice,:]=np.where((Mat_area[id_l_Matrice,:]==Max_area) & (Mat_area[id_l_Matrice,:]!=0),Max_area,0)
            if np.sum(Mat_area[id_l_Matrice,:])>np.max(Mat_area[id_l_Matrice,:]): #rare case when a mother cell have multiple daughters cells withs same areas (one pixel for example)
                Mat_area[id_l_Matrice,:]=np.where((Mat_area[id_l_Matrice,:]==np.max(Mat_area[id_l_Matrice,:])) & (Actual==np.max(Actual[Mat_area[id_l_Matrice,:]==np.max(Mat_area[id_l_Matrice,:])])),np.max(Mat_area[id_l_Matrice,:]),0)  #We take the number of the newest mother (biggest id)

        if np.max(Prev)>prev_max:
            prev_max=int(np.max(Prev))     
            
        #Columns
        for id_c_Matrice in range(len(Mat_area[0,:])):
            Matrice[:,id_c_Matrice]=np.where((Mat_area[:,id_c_Matrice]==np.max(Mat_area[:,id_c_Matrice])) & (Mat_area[:,id_c_Matrice]!=0),1,0)
            S=np.sum(Matrice[:,id_c_Matrice])

            if S==1:
                New_Actual[id_c_Matrice]=Prev[Matrice[:,id_c_Matrice]==1]
            if S==0:
                New_Actual[id_c_Matrice]=prev_max+1
                prev_max=New_Actual[id_c_Matrice]
           
            if S>1:  #rare case when a daughter cell have multiple mothers cells withs same areas (one pixel for example)
                New_Actual[id_c_Matrice]=np.max(Prev[Matrice[:,id_c_Matrice]==1]) #We take the number of the newest mother (biggest id)
        for i in range(len(Actual)):
            Cl_clean[(T==Period[id_period])&(Cl==Actual[i])]=New_Actual[i]
        Cells[:,1]=Cl_clean
    
    Cells[:,1]=Cl_clean # NEW CLUSTERS ID
    
    for i in range(len(Cells[:,2])):     #Convert distance into lat and lon 
        Cells[i,2],Cells[i,3]=m(Cells[i,2],Cells[i,3],inverse=True)
        Cells[i,4],Cells[i,5]=m(Cells[i,4],Cells[i,5],inverse=True)
        Cells[i,9],Cells[i,10]=m(Cells[i,9],Cells[i,10],inverse=True)
    
    ######## SAVING CELLS file #################"
    np.save(Path(Wdir,DAY,'ECTA','ECTA_CELL_data_'+domain+'_'+str(int(Delta_D/1000))+'km.npy'), Cells) #saving the Array CELLS with all the borders and cells
    ########################################################
    
    ################## STATS CELLS ###############
    
    nb_cells=len(np.unique(Cells[:,1]).astype(int)) 
    CELL_ID=np.unique(Cells[:,1]).astype(int)
    
    run_end=time.time()
    Run_time_list=np.append(Run_time_list,run_end-run_start)
    

    m2 = Basemap(projection='merc',resolution='h',fix_aspect=True,suppress_ticks=False,llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max,lat_ts=20)
    
    print('STATS on cells --- %s seconds --- '% (time.time() - start_time))
    Cells_duration=np.zeros(nb_cells)
    Cells_area_max=np.zeros(nb_cells)
    Cells_area_mean=np.zeros(nb_cells)
    Cells_speed=np.zeros(nb_cells)
    c=0
    for i in CELL_ID:
        #print(i)
        unique,index=np.unique(Cells[Cells[:,1]==i][:,0],return_index=True)
        Cells_duration[c]=Cells[Cells[:,1]==i][:,0][-1]-Cells[Cells[:,1]==i][:,0][0] #in minutes (can have period without flash) 
        Cells_area_mean[c]=np.mean(Cells[Cells[:,1]==i][:,7][index])
        Cells_area_max[c]=np.max(Cells[Cells[:,1]==i][:,7])  #take area at each time needed
        if Cells_duration[c]>0:
            Xmax,Ymax=m2(np.max(Cells[Cells[:,1]==i][:,4]),np.max(Cells[Cells[:,1]==i][:,5]))
            Xmin,Ymin=m2(np.min(Cells[Cells[:,1]==i][:,4]),np.min(Cells[Cells[:,1]==i][:,5]))
            Cells_speed[c]=(np.sqrt((Xmax-Xmin)**2+(Ymax-Ymin)**2)/Cells_duration[c])*60/1000 
        if Cells_duration[c]==0:
            Cells_speed[c]=0
        
        c=c+1
    
    Cells_Duration_list.append(Cells_duration)
    Cells_speed_list.append(Cells_speed)
    
 
    # Choosing only cells that last more than 20 minutes 
    CELL_ID_filtered=CELL_ID[Cells_duration>20]

    nb_cells_filtered=len(CELL_ID_filtered)
    
    if nb_cells_filtered==0: #no data in the domain for the day 
        np.save(Path(Wdir,DAY,'ECTA','ECTA_CELL_data_'+domain+'_10km.npy'), -999)
        print('NO CELLS>20 min \n')
        return Continuous_activity
    
    
    ###################SUMMARY PLOT OF FILTERED CLUSTERS TRAJECTORIES CARTOPY#######
    if len(CELL_ID_filtered)!=0:
        print('Plot all cells with duration > 20 min --- %s seconds --- '% (time.time() - start_time))
  
        fig= plt.figure(figsize=(7.5,7.5))
        ax = fig.add_subplot(111,projection=ccrs.Mercator())
        ax.set_extent([lon_min,lon_max,lat_min,lat_max])
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))
        ax.add_feature(cfeature.OCEAN.with_scale('10m'))
        ax.add_feature(cfeature.STATES.with_scale('10m'),linestyle='--', alpha=.5)
        gl=ax.gridlines(draw_labels=True,linestyle='--',color='gray',alpha=0.5)
        gl.xlabels_top=False
        gl.ylabels_right=False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        Period_filtered_cells=np.unique(Cells[np.isin(Cells[:,1],CELL_ID_filtered)][:,0])
        Time_filtered_cells=TIME[np.isin(Period,Period_filtered_cells)]
        Time_min=np.min(Time_filtered_cells)
        Time_max=np.max(Time_filtered_cells)
        norm = matplotlib.colors.Normalize(vmin=Time_min, vmax=Time_max)
        cmap_time= matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)
        for cell_id in np.unique(Cells[np.isin(Cells[:,1],CELL_ID_filtered)][:,1]).astype(int): 
            print("Cell ID:",cell_id)
            unique,index=np.unique(Cells[Cells[:,1]==cell_id][:,0], return_index=True)
            Cell_trajectory_x=Cells[Cells[:,1]==cell_id][:,4][index]
            Cell_trajectory_y=Cells[Cells[:,1]==cell_id][:,5][index]
            Cell_TIME=TIME[np.isin(Period,unique)]
            ax.plot(Cell_trajectory_x[::5],Cell_trajectory_y[::5],c='black',linewidth=1,transform=ccrs.PlateCarree(),alpha=1,zorder=1)
            p=ax.scatter(Cell_trajectory_x[::5],Cell_trajectory_y[::5],norm=norm,c=Cell_TIME[::5],s=10,marker="o",cmap=cmap_jet,transform=ccrs.PlateCarree(),zorder=2)
            ax.scatter(Cell_trajectory_x[-1],Cell_trajectory_y[-1],norm=norm,s=40,color=cmap_time.to_rgba(Cell_TIME[-1]),marker="X",transform=ccrs.PlateCarree(),zorder=2)
            ax.text(Cell_trajectory_x[0]-0.1,Cell_trajectory_y[0]-0.075,str(cell_id),fontsize=10,fontweight='bold',color='black',transform=ccrs.PlateCarree(),zorder=3)
        plt.title('ECTA filtered cells trajectories-'+DAY, pad=20) 
        divider = make_axes_locatable(ax)
        ax_cb=divider.append_axes("right", size="5%", pad="2%",axes_class=plt.Axes)
        fig.add_axes(ax_cb)
        cbar=plt.colorbar(p,cax=ax_cb,label='Time (UTC)',format=DateFormatter('%H:%M'),ticks=matplotlib.dates.MinuteLocator(byminute=range(0,60,30)))
        cbar.ax.minorticks_on()
        plt.tight_layout()
        plt.savefig(Path(Wdir,DAY,'ECTA','NEW_Filtered_cells_trajectories_'+domain+'_'+str(THLD)+'_THLD.jpeg'),dpi=300,bbox_inches="tight") 
        plt.close() 
        
        
    if DAY=='180726': #Plot clusters for the 180726 situation between 13 and 14 UTC
        for i in Period: #loop on all time with activity but FED is every minutes
            if (i>=13*60) & (i<14*60):
                print(i)
                fig= plt.figure(figsize=(7.5,7.5))
                ax = fig.add_subplot(111,projection=ccrs.Mercator())
                ax.set_extent([8.4, 9.6, 41.4, 43])
                #ax.set_extent([lon_min,lon_max,lat_min,lat_max])
                ax.coastlines(resolution='10m')
                ax.add_feature(cfeature.LAND.with_scale('10m'))
                ax.add_feature(cfeature.OCEAN.with_scale('10m'))
                ax.add_feature(cfeature.STATES.with_scale('10m'),linestyle='--', alpha=.5)
                gl=ax.gridlines(draw_labels=True,linestyle='--',color='gray',alpha=0.5)
                gl.xlabels_top=False
                gl.ylabels_right=False
                #gl.xlocator = mticker.FixedLocator(np.arange(8.4,10,0.3))
                #gl.ylocator = mticker.FixedLocator([-180, -45, 0, 45, 180]
                gl.xformatter = LONGITUDE_FORMATTER
                gl.yformatter = LATITUDE_FORMATTER
                i2=np.where(DeltaT==i)[0][0]
                Bounds_FED=[1,2,3,4,5,6,7,8,9,10,20,30,50,75,100]
                norm_FED=matplotlib.colors.BoundaryNorm(Bounds_FED,cmap_density.N)
                p=ax.pcolormesh(LON2D,LAT2D,np.ma.masked_where(FED_TIME[:,:,i2].T==0,FED_TIME[:,:,i2].T),cmap=cmap_density,norm=norm_FED,transform=ccrs.PlateCarree(),zorder=2)
            
        
                plt.title(DAY+'-'+Convert_time(i-Time_interval)+ r'$\rightarrow$'+Convert_time(i)+' UTC',pad=20) 
            
                i2=i2+1
                for j in np.unique(Cells[Cells[:,0]==i][:,1]):
                    ax.plot(Cells[(Cells[:,0]==i)&(Cells[:,1]==j)][:,2],Cells[(Cells[:,0]==i)&(Cells[:,1]==j)][:,3],color='black',linewidth=1,transform=ccrs.PlateCarree(),zorder=2)
                    #ax.text(Cells[(Cells[:,0]==i)&(Cells[:,1]==j)][0,4]+0.05,Cells[(Cells[:,0]==i)&(Cells[:,1]==j)][0,5]+0.05,s=str(int(j)),fontweight='bold',transform=ccrs.PlateCarree()) #First element of x and y centroid column bcs all the same                  
                    ax.text(np.max(Cells[(Cells[:,0]==i)&(Cells[:,1]==j)][:,2])+0.0075,np.max(Cells[(Cells[:,0]==i)&(Cells[:,1]==j)][0,5])+0.04,s=str(int(j)),fontweight='bold',transform=ccrs.PlateCarree(),zorder=3,fontsize=15)
                divider = make_axes_locatable(ax)
                ax_cb=divider.append_axes("right", size="5%", pad="2%",axes_class=plt.Axes)
                fig.add_axes(ax_cb)
                plt.colorbar(p,cax=ax_cb,ticks=Bounds_FED,label='FED [#f.$km^{-2}.5min^{-1}$]',extend='max')
                #plt.colorbar(p,ticks=Bounds_FED,label='FED [#f.$km^{-2}.5min^{-1}$]',extend='max',fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(Path(Wdir,DAY,'ECTA','NEW_Flash_density_'+domain+'_'+str(int(Delta_D/1000))+'km_'+str(int(i))+'.jpeg'),dpi=300,bbox_inches="tight") 
                plt.close()
        
        
    print("END --- %s seconds ---" % (time.time() - start_time))
    return Continuous_activity
