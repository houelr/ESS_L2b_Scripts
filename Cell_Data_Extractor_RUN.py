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
from matplotlib.dates import num2date
from shapely.geometry import Polygon
from shapely.geometry import Point
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import netCDF4 as nc
from matplotlib.path import Path as mPath
from pathlib import Path
start_time = time.time()

#some functions used 

def Convert_time_UTC(time):
      A=matplotlib.dates.num2date(time)
      H="%02d" % A.hour+"%02d" % A.minute
      return H 
  


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

########################## PARAMETERS AND LMA SELECTION ##########################
#size for plot 
s_CG=100
s_VHF=2
s_IC=30
coeff=3 #multiple of s for flash plot
H_max=15

def Cell_Data_Extractor(DAY,domain,LMA,Wdir):
    if LMA=='SAETTA':  #GRID for FED and rain around Corsica
        #GRID LOADING 
        GRID_LATLON2D_CORSICA_1km=np.load(Path(Wdir,'GRID_LATLON2D_CORSICA_1km.npz')) #my 1 km*2 grid over Corsica in LAT/LON
        LON2D=GRID_LATLON2D_CORSICA_1km['LON2D']  #to get that, i did meshgrid with XD and YD --> 2D array in meter and then m(X2D,Y2D)
        LAT2D=GRID_LATLON2D_CORSICA_1km['LAT2D']
        GRID_XY2D_CORSICA_1km=np.load(Path(Wdir,'GRID_XY2D_CORSICA_1km.npz'))  #my 1 km*2 grid over Corsica in teh basemap projection (meter)
        XD=GRID_XY2D_CORSICA_1km['XD']
        YD=GRID_XY2D_CORSICA_1km['YD']
        XDT=GRID_XY2D_CORSICA_1km['XDT']
        YDT=GRID_XY2D_CORSICA_1km['YDT']
        
        lat_min_REGION=41 
        lat_max_REGION=43.5
        lon_min_REGION=6.5
        lon_max_REGION=10.6

    print('START CELL EXTRACTOR: FLASHES STATS, CHARGE LAYER RETRIEVAL (CHARGEPOL), FOR EACH CELL')
    print(DAY)
    Cells=np.load(Path(Wdir,DAY,'ECTA','ECTA_CELL_data_'+domain+'_10km.npy'))
    if np.any(Cells==-999): #no cells 
        print('NO cell for the DAY')
        np.savez(Path(Wdir,DAY,'ECTA'+domain+'_Cells_ID_list'),CELLS_ID_list=-999)
        return
    CELL_ID=np.unique(Cells[:,1]).astype(int)
    Cells_duration=np.zeros(len(CELL_ID))   
    c=0
    for i in CELL_ID:
        Cells_duration[c]=Cells[Cells[:,1]==i][:,0][-1]-Cells[Cells[:,1]==i][:,0][0] #in minutes 
        c=c+1
    CELLS_ID_filtered=CELL_ID[Cells_duration>20]
    
    print('Number of cells: '+str(len(CELLS_ID_filtered)))
    print('List of cells:', CELLS_ID_filtered)
    #FED for this cell
    m = Basemap(projection='merc',resolution='h',fix_aspect=True,llcrnrlat=lat_min_REGION,urcrnrlat=lat_max_REGION,llcrnrlon=lon_min_REGION,urcrnrlon=lon_max_REGION)
   
    date_DAY=datetime(2000+int(DAY[0:2]),int(DAY[2:4]),int(DAY[4:6]))
    Year=str(date_DAY.year)
    Month=str("%02d" % date_DAY.month)
    Day=str("%02d" % date_DAY.day)
    c1=0
    
    for Cell_id in CELLS_ID_filtered:
        print('\n'+DAY+ ' - CELL: '+str(Cell_id)+' ('+str(np.round(np.where(CELLS_ID_filtered==Cell_id)[0][0]/len(CELLS_ID_filtered)*100,1))+'%)')
        Paths=[]
        #LMA
        Big_LMA_time=np.empty(0)
        Big_LMA_time_period=np.empty(0)
        Big_LMA_lat_tmp=np.empty(0) 
        Big_LMA_lon_tmp=np.empty(0) 
        Big_LMA_nb_station=np.empty(0) 
        Big_LMA_chi2=np.empty(0)   
        Big_LMA_alt=np.empty(0) 
        Big_LMA_power=np.empty(0) 
        Big_LMA_flash=np.empty(0) 
        
        #MET
        Big_MET_time=np.empty(0)
        Big_MET_time_period=np.empty(0)
        Big_MET_strk_to_LMA=np.empty(0)
        Big_MET_lat_tmp=np.empty(0)
        Big_MET_lon_tmp=np.empty(0)
        Big_MET_current=np.empty(0)
        Big_MET_type=np.empty(0)
        Big_MET_quality=np.empty(0)
        c=0
        A_nb_flash=0
   
        t_min=int(np.min(Cells[Cells[:,1]==Cell_id][:,0])/60)
        t_max=int(np.max(Cells[Cells[:,1]==Cell_id][:,0])/60)
        print('H_min:',t_min,'H_max:',t_max)
        
        #Cells  BE careful flash can be outside the limit !!!!! it  s why  i add 1°
        lon_cell_domain_min=np.min(Cells[Cells[:,1]==Cell_id][:,2])-1
        lon_cell_domain_max=np.max(Cells[Cells[:,1]==Cell_id][:,2])+1
        lat_cell_domain_min=np.min(Cells[Cells[:,1]==Cell_id][:,3])-1
        lat_cell_domain_max=np.max(Cells[Cells[:,1]==Cell_id][:,3])+1
            
                
       #L2b DATA
        if LMA=='SAETTA':
            for i in np.arange(t_min, t_max+1):
                if i<10:
                    #Paths.append(sorted(glob.glob(str(Path(Wdir,DAY,'L2b_V02.MTRG_'+Year+'-'+Month+'-'+Day+'_formated2.merged_with_SAETTA.L2.LYLOUT_'+DAY+'_0'+str(i)+'*')))))
                    Paths.append(sorted(glob.glob(str(Path(Wdir,DAY,'L2b.V02.EXAEDRE.SAETTA.MTRG.'+Year+Month+Day+'_0'+str(i)+'*')))))
                if i>=10:
                    #Paths.append(sorted(glob.glob(str(Path(Wdir,DAY,'L2b_V02.MTRG_'+Year+'-'+Month+'-'+Day+'_formated2.merged_with_SAETTA.L2.LYLOUT_'+DAY+'_'+str(i)+'*')))))
                    Paths.append(sorted(glob.glob(str(Path(Wdir,DAY,'L2b.V02.EXAEDRE.SAETTA.MTRG.'+Year+Month+Day+'_'+str(i)+'*')))))
            if t_min>=24 or t_max>=24 :
                A=t_max-24
                print('Evening late activity, loading next day files')
                date_next_DAY=date_DAY+ timedelta(days=1)
                N_Date_full_formated=str(date_next_DAY.year)+str("%02d" % date_next_DAY.month)+str("%02d" % date_next_DAY.day)
                if os.path.exists(Wdir+'/'+N_Date_full_formated):
                   #N for next 
                   N_Year=str(date_next_DAY.year)
                   N_Month=str("%02d" % date_next_DAY.month)
                   N_Day=str("%02d" % date_next_DAY.day)
                   for i in np.arange(0,A+1):  
                       #Paths.append(sorted(glob.glob(str(Path(Wdir,N_Date_full_formated,'L2b_V02.MTRG_'+N_Year+'-'+N_Month+'-'+N_Day+'_formated2.merged_with_SAETTA.L2.LYLOUT_'+N_Date_full_formated+'_0'+str(i)+'*')))))
                       Paths.append(sorted(glob.glob(str(Path(Wdir,N_Date_full_formated,'L2b.V02.EXAEDRE.SAETTA.MTRG.'+N_Year+N_Month+N_Day+'_0'+str(i)+'*')))))
                else:
                    print('NO FILES FOR THE NEXT DAY') 
                           
                
        PATHS = [item for sublist in Paths for item in sublist]
        print('L2B Data')
        #print(PATHS)
        for files_path in PATHS:  
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
            LMA_power=LMA_data[5,:]
            LMA_nb_station=LMA_data[7,:]
            LMA_flash=LMA_data[8,:].astype(np.int64)
          
            Big_LMA_time=np.append(Big_LMA_time,LMA_time)
            Big_LMA_time_period=np.append(Big_LMA_time_period,LMA_time_period)
            Big_LMA_lat_tmp=np.append(Big_LMA_lat_tmp,LMA_lat_tmp)
            Big_LMA_lon_tmp=np.append(Big_LMA_lon_tmp,LMA_lon_tmp)
            Big_LMA_nb_station=np.append(Big_LMA_nb_station,LMA_nb_station)
            Big_LMA_chi2=np.append(Big_LMA_chi2,LMA_chi2)
            Big_LMA_alt=np.append(Big_LMA_alt,LMA_alt)
            Big_LMA_power=np.append(Big_LMA_power,LMA_power)
        
            c=np.max(LMA_flash)+1
            
            ##METEORAGE
            MET_data=ds_L2b['meteorage_data'][:,:].data
            if MET_data.size==0:
                print('NO MET DATA !')
                if A_nb_flash==0:
                    N=np.max(LMA_flash)+1    
                N=N+c    #Number max of flash in this period  
                for f in (np.unique(LMA_flash)): #Total Flash + max period to avoid problems 
                    LMA_flash[LMA_flash==f]=N 
                    N=N+1   
                A_nb=len(np.unique(LMA_flash))
                A_nb_flash=A_nb_flash+A_nb
                Big_LMA_flash=np.append(Big_LMA_flash,LMA_flash)
                continue
            MET_time=ds_L2b['timus_meteorage'][:].data-366
            MET_strk_to_LMA=ds_L2b['meteorage_stroke_to_saetta_flash'][:].data.astype(int)
            #MET_strk_to_LMA=np.squeeze( MET_strk_to_LMA,axis=1)
            MET_time_period=(MET_time-START_DAY)*24.*60. 
            MET_lat=MET_data[6,:]
            MET_lon=MET_data[7,:]
            MET_current=MET_data[8,:]
            MET_type=MET_data[9,:]
            MET_quality=MET_data[10,:]
            Big_MET_time=np.append(Big_MET_time,MET_time)
            Big_MET_time_period=np.append(Big_MET_time_period,MET_time_period)
            Big_MET_lat_tmp=np.append(Big_MET_lat_tmp,MET_lat)
            Big_MET_lon_tmp=np.append(Big_MET_lon_tmp,MET_lon)
            Big_MET_current=np.append(Big_MET_current,MET_current)
            Big_MET_type=np.append(Big_MET_type,MET_type)
            Big_MET_quality=np.append(Big_MET_quality,MET_quality)
            if A_nb_flash==0:
                N=np.max(LMA_flash)+1       #Number max of flash in this period  
            N=N+c
            for f in (np.unique(LMA_flash)): #Total Flash + max period to avoid problems 
                MET_strk_to_LMA[MET_strk_to_LMA==f]=N
                LMA_flash[LMA_flash==f]=N 
                N=N+1   
            A_nb=len(np.unique(LMA_flash))
            A_nb_flash=A_nb_flash+A_nb   
            Big_LMA_flash=np.append(Big_LMA_flash,LMA_flash)
            Big_MET_strk_to_LMA=np.append(Big_MET_strk_to_LMA,MET_strk_to_LMA)
            N=np.max(Big_LMA_flash)+1 

        print(len(Big_LMA_alt))
        print(len(Big_MET_current))
        print("DATA TREATMENT --- %s seconds ---" % (time.time() - start_time))
          
        ################################################### LMA Treatment ########################################"""
        #Data only in a smaller domain around the cell  
        Big_LMA_lon=Big_LMA_lon_tmp[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_lat=Big_LMA_lat_tmp[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_time=Big_LMA_time[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_nb_station=Big_LMA_nb_station[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_chi2=Big_LMA_chi2[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_time_period=Big_LMA_time_period[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_alt=Big_LMA_alt[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_power=Big_LMA_power[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        Big_LMA_flash=Big_LMA_flash[(Big_LMA_lon_tmp>lon_cell_domain_min) & (Big_LMA_lon_tmp<lon_cell_domain_max) & (Big_LMA_lat_tmp>lat_cell_domain_min) & (Big_LMA_lat_tmp<lat_cell_domain_max)]
        
        
        # Sources detected by less than 7 stations removed
        nb_stations_filter=7
        Big_LMA_time=Big_LMA_time[Big_LMA_nb_station>= nb_stations_filter]
        Big_LMA_time_period=Big_LMA_time_period[Big_LMA_nb_station>= nb_stations_filter]
        Big_LMA_lat=Big_LMA_lat[Big_LMA_nb_station>= nb_stations_filter] 
        Big_LMA_lon=Big_LMA_lon[Big_LMA_nb_station>= nb_stations_filter]
        Big_LMA_chi2=Big_LMA_chi2[Big_LMA_nb_station>= nb_stations_filter]
        Big_LMA_alt=Big_LMA_alt[Big_LMA_nb_station>= nb_stations_filter]
        Big_LMA_power=Big_LMA_power[Big_LMA_nb_station>= nb_stations_filter]
        Big_LMA_flash=Big_LMA_flash[Big_LMA_nb_station>= nb_stations_filter]
        

        # Sources with chi2>0.5 removed
        chi2_filter=0.5
        Big_LMA_time=Big_LMA_time[Big_LMA_chi2<chi2_filter]
        Big_LMA_time_period=Big_LMA_time_period[Big_LMA_chi2<chi2_filter]
        Big_LMA_lat=Big_LMA_lat[Big_LMA_chi2<chi2_filter]
        Big_LMA_lon=Big_LMA_lon[Big_LMA_chi2<chi2_filter]
        Big_LMA_alt=Big_LMA_alt[Big_LMA_chi2<chi2_filter]
        Big_LMA_power=Big_LMA_power[Big_LMA_chi2<chi2_filter]
        Big_LMA_flash=Big_LMA_flash[Big_LMA_chi2<chi2_filter]
        
        TH_sources=10 #Threshold for sources for number minimum of sources per flash #10
        #Flashes with less than 11 sources removed  (flash with more than 10 sources so 10 mini)
        unique, counts = np.unique(Big_LMA_flash, return_counts=True)  
        Long_flashes=unique[counts>=TH_sources]    
        Big_LMA_long_flash_condition=np.isin(Big_LMA_flash,Long_flashes) 
        Big_LMA_time=Big_LMA_time[Big_LMA_long_flash_condition]
        Big_LMA_time_period=Big_LMA_time_period[Big_LMA_long_flash_condition]
        Big_LMA_lat=Big_LMA_lat[Big_LMA_long_flash_condition]
        Big_LMA_lon=Big_LMA_lon[Big_LMA_long_flash_condition]
        Big_LMA_flash=Big_LMA_flash[Big_LMA_long_flash_condition]
        Big_LMA_alt=Big_LMA_alt[Big_LMA_long_flash_condition]
        Big_LMA_power=Big_LMA_power[Big_LMA_long_flash_condition]
    
        
        ################################################################ MET Treatment ######################################
        #Data only in a smaller domain around the cell
        Big_MET_lon=Big_MET_lon_tmp[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_lat=Big_MET_lat_tmp[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_time=Big_MET_time[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_time_period=Big_MET_time_period[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_strk_to_LMA=Big_MET_strk_to_LMA[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_current=Big_MET_current[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_type=Big_MET_type[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        Big_MET_quality=Big_MET_quality[(Big_MET_lon_tmp>lon_cell_domain_min) & (Big_MET_lon_tmp<lon_cell_domain_max) & (Big_MET_lat_tmp>lat_cell_domain_min) & (Big_MET_lat_tmp<lat_cell_domain_max)]
        
        
        #Weak CGs become ICs and Strong ICs become CGs
        AP_CG=len(Big_MET_current[(Big_MET_type==0) & (Big_MET_current>0)])
        AN_CG=len(Big_MET_current[(Big_MET_type==0) & (Big_MET_current<0)])
        AP_IC=len(Big_MET_current[(Big_MET_type==1) & (Big_MET_current>0)])
        AN_IC=len(Big_MET_current[(Big_MET_type==1) & (Big_MET_current<0)])
        
        Big_MET_type[(Big_MET_type==0) & (Big_MET_current<10) & (Big_MET_current>0)]=1 # weak +cg<10 kA become +IC  
        Big_MET_type[(Big_MET_type==0) & (Big_MET_current>-10) & (Big_MET_current<0)]=1 # weak -cg>-10 kA become -IC  
        Big_MET_type[(Big_MET_type==1) & ((Big_MET_current>25) | (Big_MET_current<-25))]=0 #strong abs(IC) >25 kA become CG
        
        
        BP_CG=len(Big_MET_current[(Big_MET_type==0) & (Big_MET_current>0)])
        BN_CG=len(Big_MET_current[(Big_MET_type==0) & (Big_MET_current<0)])
        BP_IC=len(Big_MET_current[(Big_MET_type==1) & (Big_MET_current>0)])
        BN_IC=len(Big_MET_current[(Big_MET_type==1) & (Big_MET_current<0)])
        if (AP_CG!=0) & (AN_CG!=0) & (AP_IC!=0) & (AN_IC!=0):
             print('Reclassification: '+str(np.round((AP_CG-BP_CG)/AP_CG*100,2))+' % + strokes deleted')
             print('Reclassification: '+str(np.round((AN_CG-BN_CG)/AN_CG*100,2))+' % - strokes deleted')
             print('Reclassification: '+str(np.round((AP_IC-BP_IC)/AP_IC*100,2))+' % + pulses deleted')
             print('Reclassification: '+str(np.round((AN_IC-BN_IC)/AN_IC*100,2))+' % - pulses deleted')
             
             
        #################################################### EXTRACTION OF FLASHES in the cell ################################################
        Cell_LMA_time=np.empty(0)   
        Cell_LMA_lat=np.empty(0) 
        Cell_LMA_lon=np.empty(0) 
        Cell_LMA_alt=np.empty(0)
        Cell_LMA_power=np.empty(0)
        Cell_LMA_flash=np.empty(0)
        Cell_LMA_time_period=np.empty(0)
        
        Cell_MET_time=np.empty(0)
        Cell_MET_lat=np.empty(0)
        Cell_MET_lon=np.empty(0)
        Cell_MET_type=np.empty(0)
        Cell_MET_strk_to_LMA=np.empty(0)
        Cell_MET_current=np.empty(0)
        Cell_MET_quality=np.empty(0)
        
        def contains_py(array, poly):
            return np.array([poly.contains(p) for p in array])
        
        print("Flashes in polygon EXTRACTION --- %s seconds ---" % (time.time() - start_time))
        for period in np.unique(Cells[Cells[:,1]==Cell_id][:,0]).astype(int):     #LOOP on the time intervals (one minute)
            XX=Cells[(Cells[:,0]==period) & (Cells[:,1]==Cell_id)][:,2]   #cells borders
            YY=Cells[(Cells[:,0]==period) & (Cells[:,1]==Cell_id)][:,3]     #cells borders
            XY=list(tuple(np.array((XX,YY)).T)) 
            poly=Polygon(XY)    #polygon object
         
            #LMA  # filter on the interval of 1 min
            Interval_LMA_condition=(Big_LMA_time_period>period-1)&(Big_LMA_time_period<=period)
            LON_LMA_P=Big_LMA_lon[Interval_LMA_condition]
            LAT_LMA_P=Big_LMA_lat[Interval_LMA_condition]
            TIME_LMA_P=Big_LMA_time[Interval_LMA_condition]
            ALT_LMA_P=Big_LMA_alt[ Interval_LMA_condition]
            POWER_LMA_P=Big_LMA_power[Interval_LMA_condition]
            FLASH_LMA_P=Big_LMA_flash[Interval_LMA_condition]
            TIME_PERIOD_LMA_P=Big_LMA_time_period[Interval_LMA_condition]
            
            #MET # filter on the interval of 1 min
            Interval_MET_condition=(Big_MET_time_period>period-1)&(Big_MET_time_period<=period)
            LON_MET_P=Big_MET_lon[Interval_MET_condition]
            LAT_MET_P=Big_MET_lat[Interval_MET_condition]
            TIME_MET_P=Big_MET_time[Interval_MET_condition]
            TYPE_MET_P=Big_MET_type[Interval_MET_condition]
            STRK_MET_P=Big_MET_strk_to_LMA[Interval_MET_condition]
            CURRENT_MET_P=Big_MET_current[Interval_MET_condition]  
            QUALITY_MET_P=Big_MET_quality[Interval_MET_condition]
            
            for flash_id in np.unique(FLASH_LMA_P): #loop on all the flashes of this time interval 
                LMA_point=np.array([Point(LON_LMA_P[FLASH_LMA_P==flash_id][0],LAT_LMA_P[FLASH_LMA_P==flash_id][0])],dtype=object)
                Flash_in_poly=contains_py(LMA_point, poly) #test if first source is in the polygon or not
                if Flash_in_poly==1: #1st source in the cell, flash data added to cell data 
                    #LMA
                    Cell_LMA_lon=np.append(Cell_LMA_lon,LON_LMA_P[FLASH_LMA_P==flash_id])
                    Cell_LMA_lat=np.append(Cell_LMA_lat,LAT_LMA_P[FLASH_LMA_P==flash_id])
                    Cell_LMA_time=np.append(Cell_LMA_time,TIME_LMA_P[FLASH_LMA_P==flash_id])
                    Cell_LMA_alt=np.append(Cell_LMA_alt,ALT_LMA_P[FLASH_LMA_P==flash_id])
                    Cell_LMA_power=np.append(Cell_LMA_power,POWER_LMA_P[FLASH_LMA_P==flash_id])
                    Cell_LMA_flash=np.append(Cell_LMA_flash,FLASH_LMA_P[FLASH_LMA_P==flash_id])
                    Cell_LMA_time_period=np.append(Cell_LMA_time_period,TIME_PERIOD_LMA_P[FLASH_LMA_P==flash_id])
                    #MET
                    Cell_MET_lon=np.append(Cell_MET_lon,LON_MET_P[STRK_MET_P==flash_id])
                    Cell_MET_lat=np.append(Cell_MET_lat,LAT_MET_P[STRK_MET_P==flash_id])
                    Cell_MET_time=np.append(Cell_MET_time,TIME_MET_P[STRK_MET_P==flash_id])
                    Cell_MET_type=np.append(Cell_MET_type,TYPE_MET_P[STRK_MET_P==flash_id])
                    Cell_MET_strk_to_LMA=np.append(Cell_MET_strk_to_LMA,STRK_MET_P[STRK_MET_P==flash_id])
                    Cell_MET_current=np.append(Cell_MET_current,CURRENT_MET_P[STRK_MET_P==flash_id])
                    Cell_MET_quality=np.append(Cell_MET_quality,QUALITY_MET_P[STRK_MET_P==flash_id])
     
        if len(Cell_LMA_time_period)==0:
            Cell_duration=0
        else:
            Cell_duration=int(Cell_LMA_time_period[-1]-Cell_LMA_time_period[0]) #in minutes 
        print("LMA cell duration: "+str(Cell_duration)+" min")
        if Cell_duration<=20:
            print("LMA cell duration: "+str(Cell_duration)+" min, removed from dataset")
            CELLS_ID_filtered=CELLS_ID_filtered[CELLS_ID_filtered!=Cell_id]
            print('NEW list of cells: ',CELLS_ID_filtered)
            continue
          
        if len(Cell_LMA_flash)==0: #if activity at the border of the FED domain, we can have some sources of a flash inside a polygon but not the initation of teh flash and there is a bug 
            print('No trigger flash in the domain, cell not processed and removed')
            CELLS_ID_filtered=CELLS_ID_filtered[CELLS_ID_filtered!=Cell_id]
            print('NEW list of cells: ',CELLS_ID_filtered)
            continue
        
        
        #Looks like some flash at the limit between 2 periods create some small flash (less than 11 sources removed)
        unique, index, inverse, counts = np.unique(Cell_LMA_flash,return_index=True,return_inverse=True, return_counts=True)
        Cell_LMA_flash_len=counts[inverse]
        Long_flashes=unique[counts>=TH_sources] 
        Cell_LMA_long_flash_condition=np.isin(Cell_LMA_flash,Long_flashes)
        Cell_LMA_time=Cell_LMA_time[Cell_LMA_long_flash_condition]
        Cell_LMA_time_period=Cell_LMA_time_period[Cell_LMA_long_flash_condition]
        Cell_LMA_lat=Cell_LMA_lat[Cell_LMA_long_flash_condition]
        Cell_LMA_lon=Cell_LMA_lon[Cell_LMA_long_flash_condition]
        Cell_LMA_alt=Cell_LMA_alt[Cell_LMA_long_flash_condition]
        Cell_LMA_power=Cell_LMA_power[Cell_LMA_long_flash_condition]
        Cell_LMA_flash=Cell_LMA_flash[Cell_LMA_long_flash_condition]
        Cell_LMA_flash_len=Cell_LMA_flash_len[Cell_LMA_long_flash_condition]
        
        #need to remove the MET data associated to these small flashes
        Cell_MET_long_flash_condition=np.isin(Cell_MET_strk_to_LMA,Cell_LMA_flash)
        Cell_MET_current=Cell_MET_current[Cell_MET_long_flash_condition]
        Cell_MET_lat=Cell_MET_lat[Cell_MET_long_flash_condition]
        Cell_MET_lon=Cell_MET_lon[Cell_MET_long_flash_condition]
        Cell_MET_time= Cell_MET_time[Cell_MET_long_flash_condition]
        Cell_MET_type=Cell_MET_type[Cell_MET_long_flash_condition]
        Cell_MET_strk_to_LMA=Cell_MET_strk_to_LMA[Cell_MET_long_flash_condition]
        Cell_MET_quality=Cell_MET_quality[Cell_MET_long_flash_condition]
        
        ######################################### CELL FED COMPUTATION #########################################
        print("CELL FED Computation --- %s seconds ---" % (time.time() - start_time))
        #FED FOR THE CELL 
        #Making FED TIME 
        Start_FED=matplotlib.dates.num2date(np.min(Cell_LMA_time)).replace(second=0, microsecond=0)
        End_FED=matplotlib.dates.num2date(np.max(Cell_LMA_time)).replace(second=0, microsecond=0)
        Duration_FED=(End_FED-Start_FED).seconds//60  #in minutes
        FED_TIME=np.zeros(Duration_FED+1)
        
        c=0
        for t in np.arange(1,Duration_FED+2,1):
            FED_TIME[c]=matplotlib.dates.date2num(Start_FED+timedelta(minutes=int(t)))
            c=c+1
        FED_GRID_TIME=np.zeros((len(YD),len(XD),len(FED_TIME)))  #Y and X and T 
      
        i0=0
        for i in FED_TIME:  
            #Selectioning data in the Time interval 
            LAT=Cell_LMA_lat[(Cell_LMA_time<=i) & (Cell_LMA_time>i-5/24/60)]  ##### AS RADAR, SO T is for obervation between T-5 min and T 
            LON=Cell_LMA_lon[(Cell_LMA_time<=i) & (Cell_LMA_time>i-5/24/60)]  
            x_LMA,y_LMA=m(LON,LAT) 
            Flash=Cell_LMA_flash[(Cell_LMA_time<=i) & (Cell_LMA_time>i-5/24/60)]
            FED=np.zeros((len(XD),len(YD)))
            for f in np.unique(Flash):   
                LMA_density_flash,x_bins,y_bins=np.histogram2d(x_LMA[Flash==f],y_LMA[Flash==f],[XDT,YDT])
                LMA_density_flash=np.where(LMA_density_flash>0,1,0)
                FED=FED+LMA_density_flash
            FED_GRID_TIME[:,:,i0]=FED.T
            i0=i0+1 
      
        FED_5min=np.ma.masked_where(FED_GRID_TIME==0,FED_GRID_TIME) 
        
        ###############################   TOTAL FED IN 3D ##################"
        ZD=np.arange(0,15.2,0.2)
        ZDT=np.arange(0,15.4,0.2)
        TDT=np.append(FED_TIME,matplotlib.dates.date2num(matplotlib.dates.num2date(FED_TIME[-1])+timedelta(minutes=1)))
        FED_LON_LAT=np.zeros((len(XD),len(YD))) 
        FED_LON_ALT=np.zeros((len(XD),len(ZD))) 
        FED_ALT_LAT=np.zeros((len(ZD),len(YD))) 
        FED_TIME_ALT=np.zeros((len(FED_TIME),len(ZD)))
        Cell_LMA_X,Cell_LMA_Y=m(Cell_LMA_lon,Cell_LMA_lat)
        
        for f in np.unique(Cell_LMA_flash):
            LON_LAT_density_flash,x_bins,y_bins=np.histogram2d(Cell_LMA_X[Cell_LMA_flash==f],Cell_LMA_Y[Cell_LMA_flash==f],[XDT,YDT])
            LON_LAT_density_flash=np.where(LON_LAT_density_flash>0,1,0)
            FED_LON_LAT=FED_LON_LAT+LON_LAT_density_flash
            
            LON_ALT_density_flash,x_bins,y_bins=np.histogram2d(Cell_LMA_X[Cell_LMA_flash==f],Cell_LMA_alt[Cell_LMA_flash==f]/1000.,[XDT,ZDT])
            LON_ALT_density_flash=np.where(LON_ALT_density_flash>0,1,0)
            FED_LON_ALT=FED_LON_ALT+LON_ALT_density_flash
            ALT_LAT_density_flash,x_bins,y_bins=np.histogram2d(Cell_LMA_alt[Cell_LMA_flash==f]/1000.,Cell_LMA_Y[Cell_LMA_flash==f],[ZDT,YDT])
            ALT_LAT_density_flash=np.where(ALT_LAT_density_flash>0,1,0)
            FED_ALT_LAT=FED_ALT_LAT+ALT_LAT_density_flash
            
            TIME_ALT_density_flash,x_bins,y_bins=np.histogram2d(Cell_LMA_time[Cell_LMA_flash==f],Cell_LMA_alt[Cell_LMA_flash==f]/1000,[TDT,ZDT])
            TIME_ALT_density_flash=np.where(TIME_ALT_density_flash>0,1,0)
            FED_TIME_ALT=FED_TIME_ALT+TIME_ALT_density_flash
            
        FED_LON_ALT=np.ma.masked_where(FED_LON_ALT==0,FED_LON_ALT)
        FED_ALT_LAT=np.ma.masked_where(FED_ALT_LAT==0,FED_ALT_LAT)
        FED_LON_LAT=np.ma.masked_where(FED_LON_LAT==0,FED_LON_LAT)
        FED_TIME_ALT=np.ma.masked_where(FED_TIME_ALT==0,FED_TIME_ALT)
        
        ##############################################   RADAR PIXELS INSIDE CELL (polygon) FED ###################################
        print("PIXELS inside a polygon (cell) for ACRR --- %s seconds ---" % (time.time() - start_time))
        Cell_X_indexes_extractor=np.empty(0).astype(np.int64)
        Cell_Y_indexes_extractor=np.empty(0).astype(np.int64)
        Cell_TIME_indexes_extractor=np.empty(0)
        
        for t in range(len(FED_TIME)):
            if len(np.where(FED_5min[:,:,t])[0]>0)==0:
                   continue
            #Contour of the cell for radar 
            Pixel_loc_x,Pixel_loc_y=np.where(FED_5min[:,:,t]>0)
            Pixels_2D=np.zeros((len(Pixel_loc_x),2)).astype(np.int64)
            Pixels_2D[:,0]=Pixel_loc_x
            Pixels_2D[:,1]=Pixel_loc_y
            if len(Pixel_loc_x)>2:
                try:
                    Conv_surf=ConvexHull(Pixels_2D)
                    Contour=np.zeros((len(Pixel_loc_x[Conv_surf.vertices]),2))   #get the vertices of this surface 
                    Contour[:,0]=Pixel_loc_x[Conv_surf.vertices]
                    Contour[:,1]=Pixel_loc_y[Conv_surf.vertices]
                    Poly=Polygon(list(tuple(Contour)))    #Polygon with the center of pixels
                    Borders=np.asarray(Poly.exterior.xy)
                except:  #If pixels are colinear (lign, diagonal), polygon function can t make a polygon of it 
                    Borders=np.zeros((2,5))   #5 points to make closed borders, rectangular (or square) border, 5 points to make it complete 
                    Borders[0,:]=[np.min(Pixel_loc_x)-5,np.max(Pixel_loc_x)+5,np.max(Pixel_loc_x)+5,np.min(Pixel_loc_x)-5,np.min(Pixel_loc_x)-5]
                    Borders[1,:]=[np.min(Pixel_loc_y)-5,np.max(Pixel_loc_y)+5,np.max(Pixel_loc_y)+5,np.min(Pixel_loc_y)-5,np.min(Pixel_loc_y)-5]    
                    
            if len(Pixel_loc_x)<=2:    #clusters with one or two pixels
                Borders=np.zeros((2,5))   #5 points to make closed borders, rectangular (or square) border, 5 points to make it complete 
                Borders[0,:]=[np.min(Pixel_loc_x)-5,np.max(Pixel_loc_x)+5,np.max(Pixel_loc_x)+5,np.min(Pixel_loc_x)-5,np.min(Pixel_loc_x)-5]
                Borders[1,:]=[np.min(Pixel_loc_y)-5,np.max(Pixel_loc_y)+5,np.max(Pixel_loc_y)+5,np.min(Pixel_loc_y)-5,np.min(Pixel_loc_y)-5]         
    
            x, y = np.meshgrid(np.arange(len(LON2D[:,1])), np.arange(len(LON2D[0,:]))) # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T #create a combinaison of each points of the grid
            
            p = mPath(Borders.T) # make a a polygon with data
            grid = p.contains_points(points)
            Cell_pixels_X=x[grid==True]  #indexes associated to the polygon 
            Cell_pixels_Y=y[grid==True]
        
            #ARRAY with all indexes with time associated
            Cell_X_indexes_extractor=np.append(Cell_X_indexes_extractor,Cell_pixels_X)
            Cell_Y_indexes_extractor=np.append(Cell_Y_indexes_extractor,Cell_pixels_Y)
            Cell_TIME_indexes_extractor=np.append(Cell_TIME_indexes_extractor,np.repeat(FED_TIME[t],len(Cell_pixels_X)))
        
        FED_EXTRACT_TIME=np.unique(Cell_TIME_indexes_extractor)   #sometimes, no activity during some minutes so we need to selection only time with activity           
            
        ########################### Cell trajectory computation with weighted FED #################################    
        Cell_trajectory_LON=np.zeros(len(FED_EXTRACT_TIME))
        Cell_trajectory_LAT=np.zeros(len(FED_EXTRACT_TIME))
        Cell_trajectory_2D=np.zeros((len(FED_5min[:,0,0]),len(FED_5min[0,:,0]),len(FED_EXTRACT_TIME)))
        for i in range(len(FED_EXTRACT_TIME)):
            I=np.where(FED_TIME==FED_EXTRACT_TIME[i])[0][0]
            ABS_LAT=np.abs(LAT2D[:,0]-np.ma.average(LAT2D,weights=FED_5min[:,:,I]))
            X_W_mean=ABS_LAT.argmin()
            ABS_LON=np.abs(LON2D[0,:]-np.ma.average(LON2D,weights=FED_5min[:,:,I]))
            Y_W_mean=ABS_LON.argmin()
            Cell_trajectory_2D[X_W_mean,Y_W_mean,i]=1
            Cell_trajectory_2D=np.ma.masked_where(Cell_trajectory_2D==0,Cell_trajectory_2D)
            Cell_trajectory_LON[i]=LON2D[X_W_mean,Y_W_mean]
            Cell_trajectory_LAT[i]=LAT2D[X_W_mean,Y_W_mean]  
            
                
        ##################################### FLASH ID reorganization and association with MET ########################    
        SAVE_Cell_LMA_flash=np.copy(Cell_LMA_flash)
        unique_FMET,index_FMET,inverse_FMET=np.unique(Cell_MET_strk_to_LMA,return_index=True,return_inverse=True)
        unique_FLMA,index_FLMA,inverse_FLMA=np.unique(SAVE_Cell_LMA_flash,return_index=True,return_inverse=True)
        
        #reorganize flashes number with time from 0 (first flash in the cell) to N-1
        unique,index,inverse=np.unique(Cell_LMA_flash,return_index=True,return_inverse=True)
        F=np.arange(0,len(unique),1)
        Cell_LMA_flash=F[inverse]
        Cell_MET_strk_to_LMA=F[np.isin(unique_FLMA,unique_FMET)][inverse_FMET]
      
        
      ######################################### FLASH CLASSIFICATION ##############################################
        print("Flash classification --- %s seconds ---" % (time.time() - start_time))

        Cell_Flash_class_complete=[]
        Cell_Flash_class_main=[]
        for id_f in np.unique(Cell_LMA_flash).astype(int):
            if len(Cell_MET_lat[Cell_MET_strk_to_LMA==id_f])==0:#NO MET =0
                Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'NO MET')
                Cell_Flash_class_main=np.append(Cell_Flash_class_main,'NO MET')
                continue
            if np.all(Cell_MET_type[Cell_MET_strk_to_LMA==id_f]==1):#IC
                if np.all(Cell_MET_current[Cell_MET_strk_to_LMA==id_f]>0):    #+IC
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'+IC')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'IC')
                    continue
                if np.all(Cell_MET_current[Cell_MET_strk_to_LMA==id_f]<0):    #-IC
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'-IC')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'IC')
                    continue
                Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'Dual_IC')
                Cell_Flash_class_main=np.append(Cell_Flash_class_main,'IC')
            if np.all(Cell_MET_type[Cell_MET_strk_to_LMA==id_f]==0): #CG
                if np.all(Cell_MET_current[Cell_MET_strk_to_LMA==id_f]>0):    #+CG
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'+CG_pur')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'+CG')
                    
                    continue
                if np.all(Cell_MET_current[Cell_MET_strk_to_LMA==id_f]<0):    #-CG
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'-CG_pur')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'-CG')
                    continue 
                Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'Dual_CG')
                Cell_Flash_class_main=np.append(Cell_Flash_class_main,'Ambiguous')
                continue
            if (np.any(Cell_MET_type[Cell_MET_strk_to_LMA==id_f]==0) & (np.any(Cell_MET_type[Cell_MET_strk_to_LMA==id_f]==1))): #hybrid
                if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==1)]>0): #+IC hybrid
                    if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==0)]>0): #+IC hybrid +CG
                        Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'+IC_Hybrid_+CG')
                        Cell_Flash_class_main=np.append(Cell_Flash_class_main,'+CG')
                        continue
                    if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==0)]<0): #+IC -CG
                        Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'+IC_Hybrid_-CG')
                        Cell_Flash_class_main=np.append(Cell_Flash_class_main,'-CG')
                        continue
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'+IC_Hybrid_Dual_CG')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'Ambiguous')
                    continue
                    
                if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==1)]<0): #-IC hybrid
                    if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==0)]>0): #-IC hybrid +CG
                        Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'-IC_Hybrid_+CG')
                        Cell_Flash_class_main=np.append(Cell_Flash_class_main,'+CG')
                        continue
                    if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==0)]<0): #+IC -CG
                        Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'-IC_Hybrid_-CG')
                        Cell_Flash_class_main=np.append(Cell_Flash_class_main,'-CG')
                        continue 
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'-IC_Hybrid_Dual_CG')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'Ambiguous')
                    continue
                
                if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==0)]>0): #Dual IC hybrid +CG
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'Dual_IC_Hybrid_+CG')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'+CG')
                    continue
        
                if np.all(Cell_MET_current[(Cell_MET_strk_to_LMA==id_f) & (Cell_MET_type==0)]<0): #Dual IC hybrid -CG 
                    Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'Dual_IC_Hybrid_-CG')
                    Cell_Flash_class_main=np.append(Cell_Flash_class_main,'-CG')
                    continue  #Dual IC hybrid Dual CG 
                Cell_Flash_class_complete=np.append(Cell_Flash_class_complete,'Dual_IC_Hybrid_Dual_CG')
                Cell_Flash_class_main=np.append(Cell_Flash_class_main,'Ambiguous')
                continue
         
        Flash_type_id=['NO MET','+IC','-IC','Dual_IC','Dual_CG','+CG_pur','-CG_pur','+IC_Hybrid_+CG','+IC_Hybrid_-CG','+IC_Hybrid_Dual_CG','-IC_Hybrid_+CG','-IC_Hybrid_-CG','-IC_Hybrid_Dual_CG','Dual_IC_Hybrid_+CG','Dual_IC_Hybrid_-CG','Dual_IC_Hybrid_Dual_CG']
            
        
        
        
        ################################################ Flashes properties #################################"
        
        print("Before stats of flashes --- %s seconds ---" % (time.time() - start_time))
        Tau=(5*10**-4)/(3600*24) #500 micro seconds
        
        Nb_S=np.zeros(len(F))   #Number of sources for each flash
        Duration=np.zeros(len(F))      #Duration of each flash 
        Flash_time_period=np.zeros(len(F)) #time of occurence of each flash
        Flash_first_source_flag=np.zeros(len(Cell_LMA_flash)) #1 for 1st sources for each flash 0 for others
        Horizontal_Flash_area=np.zeros(len(F)) #Area of the convexhull of the horizontal plan projection of VHF sources
        L_H=np.zeros(len(F))  #sqrt of the Area, horizontal
        Multiplicity=np.zeros(len(F))   # Number of strokes associated to each CG, 0 for no CGs 
        Max_CurrentCG=np.zeros(len(F)) # Current min or max associated to each CG stokes 
        Sum_CurrentCG=np.zeros(len(F))  #Sum of all positive or negative strokes of all CG, if only one= min or max 
        Current_1st_stroke=np.zeros(len(F)) #Current associated to the first stroke for CG, 0 fo the rest 
        Flash_time_norm=np.zeros(len(F))  #normalized according to first and last VHF sources of the cell, 0 for first flash and 1 for last 
        Flash_deltaT=np.zeros(len(F)) #Time difference between max stroke associated to a CG and the start of teh flash  
        Delta_T_stroke=np.zeros(len(F)) #Time diff between flash start and first stroke 
        Delta_T_stroke_normed=np.zeros(len(F))  #normed time diff between flash start and first stroke 
        Delta_T_stroke_pulse_normed=np.zeros(len(F)) #Time diff between first stroke and first pulse 
        Delta_T_stroke_pulse=np.zeros(len(F)) #Time diff between first stroke and first pulse 
        Flash_vertical_extension=np.zeros(len(F)) #95th -5th alt for each flash
        Flash_trigger_alt=np.zeros(len(F))  #mean altitude of the sources in the first 500 mircroseconds of the flash 
        Flash_trigger_time=np.zeros(len(F))  #mean time of the sources in the first 500 mircroseconds of the flash 
        Comparison_alt=np.zeros(len(F)) 
        Comparison_time=np.zeros(len(F))
        
        unique, index, inverse, counts = np.unique(Cell_LMA_flash,return_index=True,return_inverse=True, return_counts=True)
        T0=Cell_LMA_time[index]
        Nb_S=Cell_LMA_flash_len[index]
        
        rev_unique, rev_index, rev_inverse, rev_counts = np.unique(Cell_LMA_flash[::-1],return_index=True,return_inverse=True, return_counts=True)
        TN=Cell_LMA_time[::-1][rev_index] #récupérer la dernière source même si toutes les sources d'un écalirs ne sont pas dans l'ordre
        Duration=(TN-T0)*24*3600
        Flash_time_period=Cell_LMA_time_period[index]-np.min(Cell_LMA_time_period)
        
        #class main 
        Cell_LMA_flash_class_main=Cell_Flash_class_main[inverse]
        
        #class complete
        Cell_LMA_flash_class_complete=Cell_Flash_class_complete[inverse]
        
        #alt trigger    
        Y=np.zeros(len(Cell_LMA_time))
        T0_LMA=T0[inverse]
        Y[Cell_LMA_time-T0_LMA<Tau]=Cell_LMA_alt[Cell_LMA_time-T0_LMA<Tau]
        M=np.zeros((len(F),8))
        for i in np.arange(0,8,1): M[:,i]=Y[index+i]
        M=np.ma.masked_where(M==0,M)
        Flash_trigger_alt=np.ma.getdata(np.nanmean(M,1)/1000)
        
        Comparison_alt=Cell_LMA_alt[index]/1000
        
        #time trigger
        Z=np.zeros(len(Cell_LMA_time))
        Z[Cell_LMA_time-T0_LMA<Tau]=Cell_LMA_time[Cell_LMA_time-T0_LMA<Tau]
        M=np.zeros((len(F),8))
        for i in np.arange(0,8,1): M[:,i]=Z[index+i]
        M=np.ma.masked_where(M==0,M)
        Flash_trigger_time=np.ma.getdata(np.nanmean(M,1))
        Comparison_time=Cell_LMA_time[index]
        
        Flash_time_norm=(Flash_trigger_time-Cell_LMA_time[0])/(Cell_LMA_time[-1]-Cell_LMA_time[0])
        Flash_first_source_flag=np.zeros(len(Cell_LMA_time))
        Flash_first_source_flag[index]=1
    
        c=0
        ARRAY_alt=np.zeros((len(F),np.max(counts)))
        for i in F: 
            ARRAY_alt[i,0:counts[i]]=Cell_LMA_alt[Cell_LMA_flash==i]/1000.
            if Cell_Flash_class_main[F==i]=='-CG':
                Multiplicity[c]=len(Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)])
                Max_CurrentCG[c]=np.min(Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)])
                Sum_CurrentCG[c]=np.sum(Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)])
                if len(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0) & (Cell_MET_current==Max_CurrentCG[F==i])])!=1: #case where two stroke max with same currents
                    Flash_deltaT[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0) & (Cell_MET_current==Max_CurrentCG[F==i])][0]-Flash_trigger_time[F==i])*24*3600/Duration[F==i]
                else:
                    Flash_deltaT[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0) & (Cell_MET_current==Max_CurrentCG[F==i])]-Flash_trigger_time[F==i])*24*3600/Duration[F==i]
                Delta_T_stroke[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Flash_trigger_time[F==i])*24*3600
                Delta_T_stroke_normed[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Flash_trigger_time[F==i])*24*3600/Duration[F==i]
                Current_1st_stroke[c]=Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]
                
                if len(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==1)])>0: #if there is also pulses 
                    Delta_T_stroke_pulse_normed[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==1)][0])*24*3600/Duration[F==i]
                    Delta_T_stroke_pulse[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==1)][0])*24*3600
                    
            if Cell_Flash_class_main[F==i]=='+CG':
                Multiplicity[c]=len(Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)])
                Max_CurrentCG[c]=np.max(Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)])
                Sum_CurrentCG[c]=np.sum(Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)])
                if len(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0) & (Cell_MET_current==Max_CurrentCG[F==i])])!=1: #case where two stroke max with same currents
                    Flash_deltaT[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0) & (Cell_MET_current==Max_CurrentCG[F==i])][0]-Flash_trigger_time[F==i])*24*3600/Duration[F==i]
                else:
                    Flash_deltaT[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0) & (Cell_MET_current==Max_CurrentCG[F==i])]-Flash_trigger_time[F==i])*24*3600/Duration[F==i]
                Delta_T_stroke[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Flash_trigger_time[F==i])*24*3600
                Delta_T_stroke_normed[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Flash_trigger_time[F==i])*24*3600/Duration[F==i]
                Current_1st_stroke[c]=Cell_MET_current[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]
                
                if len(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==1)])>0: #if there is also pulses 
                    Delta_T_stroke_pulse_normed[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==1)][0])*24*3600/Duration[F==i]
                    Delta_T_stroke_pulse[c]=(Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==0)][0]-Cell_MET_time[(Cell_MET_strk_to_LMA==i) & (Cell_MET_type==1)][0])*24*3600
                
                      #Horizontal flash extent 
            Flash_2D=np.zeros((len(Cell_LMA_X[Cell_LMA_flash==i]),2))
            Flash_2D[:,0]=Cell_LMA_X[Cell_LMA_flash==i]
            Flash_2D[:,1]=Cell_LMA_Y[Cell_LMA_flash==i]
            try:
                Flash_ConvHull=ConvexHull(Flash_2D)
                Horizontal_Flash_extent=Polygon(list(tuple(Flash_2D[Flash_ConvHull.vertices])))    #Polygon with the center of pixels
                #Borders=np.asarray(Flash_extent.exterior.xy)  #if we want to plot 
                Horizontal_Flash_area[c]=Horizontal_Flash_extent.area/10**6 #in km2
                L_H[c]=np.sqrt(Horizontal_Flash_area[c]) # Flash width in km 
                c=c+1
            except:
                #print('test'+str(i))
                Horizontal_Flash_area[c]=0
                L_H[c]=0 # Flash width in km 
                c=c+1
                
                
        #Flash vertical extension
        ARRAY_alt=np.where(ARRAY_alt==0,np.nan,ARRAY_alt)
        alt_95th=np.nanpercentile(ARRAY_alt,95,axis=1)
        alt_5th=np.nanpercentile(ARRAY_alt,5,axis=1)
        Flash_vertical_extension=alt_95th-alt_5th
           
    
        print('STD trigger two methods for alt :'+str(np.std(np.abs(Flash_trigger_alt-Comparison_alt)))+' km')  
        print('STD trigger two methods for time :'+str(np.std(np.abs(Flash_trigger_time-Comparison_time))*24*3600)+' s')  
        print("After stats of flashes --- %s seconds ---" % (time.time() - start_time))    
        print(len(F),'Flashes')
    
        
    
        ########################################## CELL SCALE STATS ########################## 
        Cell_duration=(matplotlib.dates.num2date(Cell_LMA_time[-1])-matplotlib.dates.num2date(Cell_LMA_time[0])).seconds//60
        Nb_flashes=len(F)
        Nb_CG=len(F[(Cell_Flash_class_main=='+CG') | (Cell_Flash_class_main=='-CG')])
        T_start=Cell_LMA_time[0]
        T_end=Cell_LMA_time[-1]
        
        ###################################################### Lightning jump ##################################
        
        Cell_duration2=int(np.max(Cell_LMA_time_period))-int(np.min(Cell_LMA_time_period))+1
        if Cell_duration2%2==1:
            Cell_duration2=Cell_duration2+1      #not odd for 2min mean
        FR_T=np.zeros(Cell_duration2)  #FR total, all flash  
        FR_PCG=np.zeros(Cell_duration2) 
        FR_NCG=np.zeros(Cell_duration2) 
        FR_PIC=np.zeros(Cell_duration2) 
        FR_NIC=np.zeros(Cell_duration2) 
        FR_DIC=np.zeros(Cell_duration2) 
        FR_NOMET=np.zeros(Cell_duration2)
        Flash_id_control=np.empty(0)
        Cell_time_minute=np.zeros(Cell_duration2)
        Cell_time_minute[0]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_LMA_time[0]).replace(second=0,microsecond=0)+timedelta(minutes=1))
        for i in np.arange(1,Cell_duration2,1):
            Cell_time_minute[i]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_time_minute[i-1])+timedelta(minutes=1))
    
        T=0
        for t in Cell_time_minute:#for plot, one minute before and after the cell, time asscoiated to the first souce for each flash 
            Time_condition_first_source=(Cell_LMA_time<=t) & (Cell_LMA_time>t-1/(24*60)) & (Flash_first_source_flag==1)
            FR_T[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source]).astype(int)) 
            FR_PCG[T]=len(np.unique(Cell_LMA_flash[(Time_condition_first_source) & (Cell_LMA_flash_class_main=='+CG')]).astype(int)) 
            FR_NCG[T]=len(np.unique(Cell_LMA_flash[(Time_condition_first_source) & (Cell_LMA_flash_class_main=='-CG')]).astype(int)) 
            FR_NIC[T]=len(np.unique(Cell_LMA_flash[(Time_condition_first_source) & (Cell_LMA_flash_class_complete=='-IC')]).astype(int)) 
            FR_PIC[T]=len(np.unique(Cell_LMA_flash[(Time_condition_first_source) & (Cell_LMA_flash_class_complete=='+IC')]).astype(int)) 
            FR_DIC[T]=len(np.unique(Cell_LMA_flash[(Time_condition_first_source) & (Cell_LMA_flash_class_complete=='Dual_IC')]).astype(int)) 
            FR_NOMET[T]=len(np.unique(Cell_LMA_flash[(Time_condition_first_source) & (Cell_LMA_flash_class_main=='NO MET')]).astype(int)) 
            Flash_id_control=np.append(Flash_id_control,np.unique(Cell_LMA_flash[Time_condition_first_source]).astype(int))
            T=T+1 
    
        FR_Tavg_2min=np.zeros(len(np.arange(0,Cell_duration2,2))) #in case of odd, the last minute not taken into account
        Time_2min=np.zeros(len(np.arange(0,Cell_duration2,2)))
        FR_Tavg_2min[0]=(FR_T[0]+FR_T[1])/2
        Time_2min[0]=Cell_time_minute[1]
        c=1
        for i in np.arange(2,Cell_duration2,2):
            FR_Tavg_2min[c]=(FR_T[i]+FR_T[i+1])/2
            Time_2min[c]=Cell_time_minute[i+1]
            c=c+1
        
        Cell_FRT_Median=np.median(FR_T)
        
        DFRDT=(FR_Tavg_2min[1::]-FR_Tavg_2min[0:-1])/2  #flashes.min-2
        LJ=np.zeros(len(np.arange(0,Cell_duration2,2)))
        DFRDT=np.append(DFRDT,0)
        
        Sigma=np.zeros(len(np.arange(0,Cell_duration2,2)))
        for i in range(len(Time_2min)):
            T=Time_2min[i]
            if len(DFRDT[(Time_2min<T) & (Time_2min>T-12/(60*24))])==5:
                Sigma[i]=np.std(DFRDT[(Time_2min<T) & (Time_2min>T-12/(60*24))])
            if (FR_Tavg_2min[i]>10) & (Sigma[i]!=0) & (DFRDT[i]>2*Sigma[i]):
                LJ[i]=1
        for i in range(len(LJ)):
            if (LJ[i]==1) & (np.sum(LJ[i-3:i+4])!=1):   #6 minutes before and after, check if multiple Lj in this interval 
                A=np.max(DFRDT[i-3:i+4][LJ[i-3:i+4]==1])
                I=np.where(DFRDT[i-3:i+4]==A)[0][0]
                LJ[i-3:i+4]=np.zeros(len(DFRDT[i-3:i+4]))
                LJ[i-3:i+4][I]=1
        
        Cell_Nb_LJ=np.sum(LJ)
        
        ###################################### Charge Layer Retrieval from CHARGEPOL (Medina et al., 2021) ##############################
        print("Before Chargepol --- %s seconds ---" % (time.time() - start_time))
        #Works well but for long flash in statiform region tilted doesn t work that well, other method with that one ? 
       
        Tau=(10*10**-3)/(3600*24)
        #AC for Automatic Charge
        F=np.unique(Cell_LMA_flash).astype(int)
        ALT_AC=np.copy(Cell_LMA_alt)/1000 #in km
        TIME_F_AC=np.copy(Cell_LMA_time)
        ACLR_Polarity_layer=np.zeros(len(ALT_AC)) 
        L_speed_list=np.zeros(len(F))
        Tresh_list=np.zeros(len(F))
        T_PB_list=np.zeros(len(F))
        Tresh_alt_10_down_list=np.zeros(len(F))
        Tresh_alt_90_down_list=np.zeros(len(F))
        Tresh_alt_10_up_list=np.zeros(len(F))
        Tresh_alt_90_up_list=np.zeros(len(F))
        Layer_thick_pos=np.zeros(len(F))
        Layer_thick_neg=np.zeros(len(F))
        Layer_height_pos=np.zeros(len(F))
        Layer_height_neg= np.zeros(len(F)) 
        MSE_list=np.zeros(len(F))
        R2_list=np.zeros(len(F))
        F_type=np.zeros(len(F))
        Len_time_window=np.zeros(len(F))
        Flag_f_ACLR=np.zeros(len(F)) #0 not used for ACLR, 1 used
        Cell_flag_flash_ACLR=np.zeros(len(Cell_LMA_time)) #0 not used for ACLR, 1 used
        c_PB=0
        Tau=(10*10**-3)/(3600*24)
        
        unique, index, inverse, counts = np.unique(Cell_LMA_flash,return_index=True,return_inverse=True, return_counts=True)
        F_S_NB=unique[counts>=20]  #Only flash with al least 20 sources
        T0=Cell_LMA_time[index] 
        T0_LMA=T0[inverse]  
        unique_S_PB,index_S_PB,inverse_S_PB,counts_S_PB =np.unique(Cell_LMA_flash[Cell_LMA_time-T0_LMA<Tau],return_index=True,return_inverse=True,return_counts=True)
        F_S_PB=unique_S_PB[counts_S_PB>=4]  #Only flashes with 4 sources mini in the PB (10 ms)
        ACLR_condition=(np.isin(unique,F_S_NB) & np.isin(unique,F_S_PB) & (Duration>0.01))  #Falshes need to last longer than PB time (10 ms) 
        F_ACLR=unique[ACLR_condition]
        T_PB_list[~ACLR_condition]=np.nan
        MSE_list[~ACLR_condition]=np.nan
        L_speed_list[~ACLR_condition]=np.nan
        R2_list[~ACLR_condition]=np.nan
        for id_f in F_ACLR.astype(int):
            T0=Cell_LMA_time[Cell_LMA_flash==id_f][0]
            T_Final=Cell_LMA_time[(Cell_LMA_time-T0<Tau) & (Cell_LMA_flash==id_f)][-1]
            T_PB=(T_Final-T0)*24*3600
            if T_PB>0.002:  #in the 10ms PB time window, need to have the fisrt and last sources spaced from minimum 2ms (avoid pocket of sources and nothing) 
                TIME_F_AC[Cell_LMA_flash==id_f]=Cell_LMA_time[Cell_LMA_flash==id_f]-T0
                TIME_WINDOW_AC=(Cell_LMA_time[(Cell_LMA_time-T0<Tau) & (Cell_LMA_flash==id_f)]-T0)*(24*3600)
                Len_time_window[F==id_f]=len(TIME_WINDOW_AC)
                c_PB=c_PB+1 #nb of flashes with 2 sources mini in 10 ms window     
                ALT_WINDOW_AC=Cell_LMA_alt[(Cell_LMA_time-T0<Tau) & (Cell_LMA_flash==id_f)]/1000.
                L_speed=np.polyfit(TIME_WINDOW_AC,ALT_WINDOW_AC,1,full=True)
                X_True=ALT_WINDOW_AC
                X_Pred=L_speed[0][0]*TIME_WINDOW_AC+L_speed[0][1]
                MSE=mean_squared_error(X_True,X_Pred)
                R2=r2_score(X_True,X_Pred)
                R2_list[F==id_f]=R2
                MSE_list[F==id_f]=MSE
                L_speed_list[F==id_f]=L_speed[0][0]
                T_PB_list[F==id_f]=T_PB
            else:
                MSE_list[F==id_f]=np.nan
                L_speed_list[F==id_f]=np.nan
                R2_list[F==id_f]=np.nan
                T_PB_list[F==id_f]=np.nan
            # vertical speed for negative leader: 5*10**4 m.s-1 so 50km.s-1
            if np.abs(L_speed_list[F==id_f])>=50 and MSE_list[F==id_f]<0.25:
                Flag_f_ACLR[F==id_f]=1
                Cell_flag_flash_ACLR[Cell_LMA_flash==id_f]=1
                if L_speed_list[F==id_f]>=50: #upward leader, positif layer above negative one, +IC configuration 
                    Tresh_H=L_speed[0][1] #origin at 0 s of the linear regression
                    Tresh_list[F==id_f]=Tresh_H
                    #10 and 90 th percentile of the data 
                    X_up = np.sort(ALT_AC[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_H)])
                    if len(X_up)>=2:  #need at least 2 sources for taking 10th and 90th percentiel sources 
                        Tresh_alt_90_up_list[F==id_f]=np.percentile(X_up,90)
                        Tresh_alt_10_up_list[F==id_f]=np.percentile(X_up,10)
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_alt_10_up_list[F==id_f]) & (ALT_AC<=Tresh_alt_90_up_list[F==id_f])]=1 #positive layer
                    else: 
                        Tresh_alt_90_up_list[F==id_f]=Tresh_H
                        Tresh_alt_10_up_list[F==id_f]=Tresh_H
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_alt_10_up_list[F==id_f])]=1 #positive layer
                        
                    X_down = np.sort(ALT_AC[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC<Tresh_H)])
                    if len(X_down)>=2:
                        Tresh_alt_90_down_list[F==id_f]=np.percentile(X_down,90)
                        Tresh_alt_10_down_list[F==id_f]=np.percentile(X_down,10)
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_alt_10_down_list[F==id_f]) & (ALT_AC<=Tresh_alt_90_down_list[F==id_f])]=-1 #negative layer
                    else: 
                        Tresh_alt_90_down_list[F==id_f]=Tresh_H
                        Tresh_alt_10_down_list[F==id_f]=Tresh_H
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC<=Tresh_alt_90_down_list[F==id_f])]=-1 #negative layer
                    
                    F_type[F==id_f]=1  #+IC==1
                    Layer_thick_pos[F==id_f]=Tresh_alt_90_up_list[F==id_f]-Tresh_alt_10_up_list[F==id_f]
                    Layer_thick_neg[F==id_f]=Tresh_alt_90_down_list[F==id_f]-Tresh_alt_10_down_list[F==id_f]
                    Layer_height_pos[F==id_f]=(Tresh_alt_90_up_list[F==id_f]+Tresh_alt_10_up_list[F==id_f])/2
                    Layer_height_neg[F==id_f]=(Tresh_alt_90_down_list[F==id_f]+Tresh_alt_10_down_list[F==id_f])/2
                       
    
                if L_speed_list[F==id_f]<=-50: # downward leader, negative layer above positive one, -IC configuration 
                    Tresh_H= L_speed[0][1] #origin at 0 s of the linear regression
                    Tresh_list[F==id_f]=Tresh_H
                    #10 and 90 th percentile of the data 
                    X_up = np.sort(ALT_AC[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_H)])
                    if len(X_up)>=2:
                        Tresh_alt_90_up_list[F==id_f]=np.percentile(X_up,90)
                        Tresh_alt_10_up_list[F==id_f]=np.percentile(X_up,10)
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_alt_10_up_list[F==id_f]) & (ALT_AC<=Tresh_alt_90_up_list[F==id_f])]=-1 #negative layer
                    else: 
                        Tresh_alt_90_up_list[F==id_f]=Tresh_H
                        Tresh_alt_10_up_list[F==id_f]=Tresh_H
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_alt_10_up_list[F==id_f])]=-1 #negative layer
                        
                    X_down = np.sort(ALT_AC[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC<Tresh_H)])
                    if len(X_down)>=2:
                        Tresh_alt_90_down_list[F==id_f]=np.percentile(X_down,90)
                        Tresh_alt_10_down_list[F==id_f]=np.percentile(X_down,10)
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC>=Tresh_alt_10_down_list[F==id_f]) & (ALT_AC<=Tresh_alt_90_down_list[F==id_f])]=1 #positive layer
                    else: 
                        Tresh_alt_90_down_list[F==id_f]=Tresh_H
                        Tresh_alt_10_down_list[F==id_f]=Tresh_H
                        ACLR_Polarity_layer[(Cell_LMA_flash==id_f) & (TIME_F_AC>Tau) & (ALT_AC<=Tresh_alt_90_down_list[F==id_f])]=1 #positive layer
                    F_type[F==id_f]=-1 #-IC=-1
                    Layer_thick_pos[F==id_f]=Tresh_alt_90_down_list[F==id_f]-Tresh_alt_10_down_list[F==id_f]
                    Layer_thick_neg[F==id_f]=Tresh_alt_90_up_list[F==id_f]-Tresh_alt_10_up_list[F==id_f]
                    Layer_height_pos[F==id_f]=(Tresh_alt_90_down_list[F==id_f]+Tresh_alt_10_down_list[F==id_f])/2
                    Layer_height_neg[F==id_f]=(Tresh_alt_90_up_list[F==id_f]+Tresh_alt_10_up_list[F==id_f])/2    

        Cell_Nb_Flash_ACLR=len(F[Flag_f_ACLR==1])
        #Number of flash of both polarity in each vertical bins (0.5 km) 
        Cell_PL=np.zeros(len(np.arange(0,H_max,0.5)))
        Cell_NL=np.zeros(len(np.arange(0,H_max,0.5)))
        for f in np.unique(F[Flag_f_ACLR==1]):
            PL,Z=np.histogram(Cell_LMA_alt[(ACLR_Polarity_layer==1) & (Cell_LMA_flash==f)]/1000.,np.arange(0,H_max+0.5,0.5))
            NL,Z=np.histogram(Cell_LMA_alt[(ACLR_Polarity_layer==-1) & (Cell_LMA_flash==f)]/1000.,np.arange(0,H_max+0.5,0.5))
            PL=np.where(PL>0,1,0)
            NL=np.where(NL>0,1,0)
            Cell_PL=Cell_PL+PL
            Cell_NL=Cell_NL+NL

       
         ####################################  10 min SAMPLES COMPUTATION  ####################################################""
        #Dominant layer polarity (bin associated to the max over the altitude) and altitude associated to 10 min bins 
        Cell_time_interval=np.zeros(Cell_duration//10+1)
        Cell_time_interval[0]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_LMA_time[0]).replace(second=0,microsecond=0)+timedelta(minutes=10))
        for i in np.arange(1,len(Cell_time_interval),1):
            Cell_time_interval[i]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_time_interval[i-1])+timedelta(minutes=10))
        Cell_time_interval[-1]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_LMA_time[-1]).replace(second=0,microsecond=0)+timedelta(minutes=1))
    
        print(len(Cell_time_interval),'bins of 10 min')
        Z=np.arange(0,H_max,0.5)
        Cell_PL_interval_max=np.zeros((len(Cell_time_interval),len(Cell_PL)))
        Cell_NL_interval_max=np.zeros((len(Cell_time_interval),len(Cell_NL)))
        Cell_PL_Sample=np.zeros((len(Cell_time_interval),len(Cell_PL)))
        Cell_NL_Sample=np.zeros((len(Cell_time_interval),len(Cell_NL)))
        Cell_Nb_F_interval=np.zeros(len(Cell_time_interval))
        Cell_Nb_F_ACLR_interval=np.zeros(len(Cell_time_interval))
        Cell_Sample_Lon=np.zeros(len(Cell_time_interval))
        Cell_Sample_Lat=np.zeros(len(Cell_time_interval))
        Cell_Sample_NCG_Multiplicity=np.zeros(len(Cell_time_interval))
        Cell_Sample_NLAYER_Area=np.zeros(len(Cell_time_interval))
        Cell_Sample_NLAYER_Horizontal_extent=np.zeros(len(Cell_time_interval))
        
        
        for i in range(0,len(Cell_time_interval),1): 
            print(Convert_time_UTC(Cell_time_interval[i]))
            if i==0:
                Time_LMA_condition=(Cell_LMA_time<Cell_time_interval[i])
                Time_FRT_condition=(Cell_time_minute<=Cell_time_interval[i]) #need to take the last minute for the 10 min period (will be the same) 
            else:
                Time_LMA_condition=(Cell_LMA_time<Cell_time_interval[i]) & (Cell_LMA_time>=Cell_time_interval[i-1])
                Time_FRT_condition=(Cell_time_minute>Cell_time_interval[i-1]) & (Cell_time_minute<=Cell_time_interval[i])
                
            LMA_flash=Cell_LMA_flash[Time_LMA_condition]
            LMA_alt=Cell_LMA_alt[Time_LMA_condition]
            ACLR_Polarity_layer_interval=ACLR_Polarity_layer[Time_LMA_condition]
            LMA_F_first_source_flag=Flash_first_source_flag[Time_LMA_condition]
            F_interval=np.unique(LMA_flash[LMA_F_first_source_flag==1])
            Flag_f_ACLR_interval=np.isin(F_interval,F[Flag_f_ACLR==1])
            if len(Multiplicity[(np.isin(F,F_interval)) & (Cell_Flash_class_main=='-CG')])!=0:
                Cell_Sample_NCG_Multiplicity[i]=np.mean(Multiplicity[(np.isin(F,F_interval)) & (Cell_Flash_class_main=='-CG')])
                
                
            NLAYER_2D=np.zeros((len(Cell_LMA_X[(Time_LMA_condition) & (ACLR_Polarity_layer==-1)]),2))
            NLAYER_2D[:,0]=Cell_LMA_X[(Time_LMA_condition) & (ACLR_Polarity_layer==-1)]
            NLAYER_2D[:,1]=Cell_LMA_Y[(Time_LMA_condition) & (ACLR_Polarity_layer==-1)]
            try:
                
                NLAYER_ConvHull=ConvexHull(NLAYER_2D)
                NLAYER_Horizontal_extent=Polygon(list(tuple(NLAYER_2D[NLAYER_ConvHull.vertices])))    #Polygon with the center of pixels
                Cell_Sample_NLAYER_Area[i]=NLAYER_Horizontal_extent.area/10**6 #in km2
                Cell_Sample_NLAYER_Horizontal_extent[i]=np.sqrt(Cell_Sample_NLAYER_Area[i]) # Flash width in km
                
            except:
               
                Cell_Sample_NLAYER_Area[i]=0
                Cell_Sample_NLAYER_Horizontal_extent[i]=0
                
            if len(FR_T[Time_FRT_condition])==0:
                print("continue")
                continue
                        
            Cell_Nb_F_interval[i]=len(F_interval)
            Cell_Nb_F_ACLR_interval[i]=len(F_interval[Flag_f_ACLR_interval==True])
            if len(LMA_flash)!=0:
                Cell_Sample_Lon[i]=np.average(Cell_LMA_lon[Time_LMA_condition])
                Cell_Sample_Lat[i]=np.average(Cell_LMA_lat[Time_LMA_condition])
            if len(LMA_flash)==0:
                Cell_Sample_Lon[i]=np.nan
                Cell_Sample_Lat[i]=np.nan
                continue
            PL_interval=np.zeros(len(Cell_PL))
            NL_interval=np.zeros(len(Cell_NL))
        
            for f in F_interval[Flag_f_ACLR_interval==True]:
                PL,Z1=np.histogram(LMA_alt[(ACLR_Polarity_layer_interval==1) & (LMA_flash==f)]/1000.,np.arange(0,H_max+0.5,0.5))
                NL,Z1=np.histogram(LMA_alt[(ACLR_Polarity_layer_interval==-1) & (LMA_flash==f)]/1000.,np.arange(0,H_max+0.5,0.5))
                PL=np.where(PL>0,1,0)
                NL=np.where(NL>0,1,0)
                PL_interval=PL_interval+PL
                NL_interval=NL_interval+NL
             
            Cell_PL_Sample[i,:]=PL_interval
            Cell_NL_Sample[i,:]=NL_interval
                
            if np.any(PL_interval)==True:  #bug if no flash (likely for the last bin)
                #max
                Cell_PL_interval_max[i,:]=np.where(PL_interval==np.max(PL_interval),1,0)
                Cell_NL_interval_max[i,:]=np.where(NL_interval==np.max(NL_interval),1,0)
        Cell_DPL_Alt_interval=np.where(Cell_PL_interval_max==1,Z,0)
        Cell_DNL_Alt_interval=np.where(Cell_NL_interval_max==1,Z,0)
        
        Samples_Polarity=np.zeros(len(Cell_time_interval))
        MIN_DPL=np.min(np.ma.masked_where(Cell_DPL_Alt_interval==0,Cell_DPL_Alt_interval),1)
        MAX_DPL=np.max(np.ma.masked_where(Cell_DPL_Alt_interval==0,Cell_DPL_Alt_interval),1)+0.5
        MIN_DNL=np.min(np.ma.masked_where(Cell_DNL_Alt_interval==0,Cell_DNL_Alt_interval),1)
        MAX_DNL=np.max(np.ma.masked_where(Cell_DNL_Alt_interval==0,Cell_DNL_Alt_interval),1)+0.5
        Samples_Polarity[(MAX_DNL<=MIN_DPL) & (MAX_DPL.mask==False)]=1  #Normal dipole, dominant positive over dominant negative
        Samples_Polarity[(MAX_DPL<=MIN_DNL) & (MAX_DPL.mask==False)]=-1   #Anomalous dipole, dominant positive below dominnat negative 
        print(Samples_Polarity,'1 for Normal, -1 for Anomalous and 0 for Unknown')    
        
        NB_SAMPLES=len(Samples_Polarity)
        NB_NP=len(Samples_Polarity[Samples_Polarity==1]) #normal
        NB_AP=len(Samples_Polarity[Samples_Polarity==-1])   #anomalous
        NB_UNK=len(Samples_Polarity[Samples_Polarity==0])   #Unknown
        if NB_SAMPLES!=0:
            print(str(np.round(NB_NP/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Normal polarity')
            print(str(np.round(NB_AP/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Anomalous polarity')
            print(str(np.round(NB_UNK/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Unknown polarity')

        Cell_Flash_R_Occup=(Nb_S*80*10**(-6)/(Duration+80*10**(-6)))*100
     
        
         
        ################################### Create directory for the cell #########################################
        dirName = domain+'_Cell_'+str(Cell_id)
        try:
            # Create target Directory
            os.mkdir(Path(Wdir,DAY,'ECTA',dirName))
            print("Directory " , dirName ,  " Created \n") 
        except FileExistsError:
             print("Directory " , dirName ,  " already exists \n")    
    
        Save_path=Path(Wdir,DAY,'ECTA',dirName)
        
        np.savez(Path(Save_path,'CELL_'+domain+'_'+str(Cell_id)),Cell_Flash_class_complete=Cell_Flash_class_complete,Cell_Flash_class_main=Cell_Flash_class_main,Flash_type_id=Flash_type_id,Cell_LMA_alt=Cell_LMA_alt, Cell_LMA_flash=Cell_LMA_flash,Cell_LMA_flash_len=Cell_LMA_flash_len,Cell_LMA_lat=Cell_LMA_lat,Cell_LMA_lon=Cell_LMA_lon,Cell_LMA_power=Cell_LMA_power,Cell_LMA_time=Cell_LMA_time,Cell_LMA_time_period=Cell_LMA_time_period,
                 Cell_MET_current=Cell_MET_current,Cell_MET_lat=Cell_MET_lat,Cell_MET_lon=Cell_MET_lon,Cell_MET_strk_to_LMA=Cell_MET_strk_to_LMA,Cell_MET_time=Cell_MET_time,Cell_MET_type=Cell_MET_type,Cell_MET_quality=Cell_MET_quality,
                 Cell_id=Cell_id,
                 FED_5min=FED_5min,FED_TIME=FED_TIME,FED_EXTRACT_TIME=FED_EXTRACT_TIME,Cell_trajectory_LON=Cell_trajectory_LON, Cell_trajectory_LAT=Cell_trajectory_LAT, Cell_trajectory_2D=Cell_trajectory_2D,
                 ZD=ZD,XD=XD,YD=YD,FED_LON_ALT=FED_LON_ALT,FED_ALT_LAT=FED_ALT_LAT,FED_LON_LAT=FED_LON_LAT,FED_TIME_ALT=FED_TIME_ALT,
                 Nb_S=Nb_S,Duration=Duration,Flash_time_period=Flash_time_period,Flash_first_source_flag= Flash_first_source_flag,Horizontal_Flash_area=Horizontal_Flash_area,
                 L_H=L_H,Multiplicity=Multiplicity,Max_CurrentCG=Max_CurrentCG,Sum_CurrentCG=Sum_CurrentCG,Current_1st_stroke=Current_1st_stroke,
                 Flash_time_norm=Flash_time_norm,Flash_deltaT=Flash_deltaT,Delta_T_stroke=Delta_T_stroke,Delta_T_stroke_normed=Delta_T_stroke_normed,
                 Delta_T_stroke_pulse_normed=Delta_T_stroke_pulse_normed,Delta_T_stroke_pulse=Delta_T_stroke_pulse,Flash_vertical_extension=Flash_vertical_extension,Cell_Nb_LJ=Cell_Nb_LJ,Cell_FRT_Median=Cell_FRT_Median,
                 Flash_trigger_alt=Flash_trigger_alt,Flash_trigger_time=Flash_trigger_time,F=F,Cell_LMA_flash_class_main=Cell_LMA_flash_class_main,Cell_LMA_flash_class_complete=Cell_LMA_flash_class_complete,Cell_Nb_Flash_ACLR=Cell_Nb_Flash_ACLR,
                 Flag_f_ACLR=Flag_f_ACLR,ACLR_Polarity_layer=ACLR_Polarity_layer,ALT_AC=ALT_AC,
                 Tresh_list=Tresh_list,L_speed_list=L_speed_list,Tresh_alt_10_up_list=Tresh_alt_10_up_list,Tresh_alt_10_down_list=Tresh_alt_10_down_list,Tresh_alt_90_down_list=Tresh_alt_90_down_list,Tresh_alt_90_up_list=Tresh_alt_90_up_list,F_type=F_type,Cell_Nb_F_interval=Cell_Nb_F_interval,Cell_Nb_F_ACLR_interval=Cell_Nb_F_ACLR_interval,
                 MSE_list=MSE_list,R2_list=R2_list,F_ACLR=F_ACLR,Len_time_window=Len_time_window,c_PB=c_PB,Layer_height_pos=Layer_height_pos,Layer_height_neg=Layer_height_neg,
                 Cell_PL=Cell_PL,Cell_NL=Cell_NL,Cell_time_interval=Cell_time_interval,Cell_DPL_Alt_interval=Cell_DPL_Alt_interval,Cell_DNL_Alt_interval=Cell_DNL_Alt_interval,Samples_Polarity=Samples_Polarity,
                 Cell_PL_interval_max=Cell_PL_interval_max,Cell_NL_interval_max=Cell_NL_interval_max,Z=Z,Cell_PL_Sample=Cell_PL_Sample,Cell_NL_Sample=Cell_NL_Sample,Cell_Sample_Lon=Cell_Sample_Lon,Cell_Sample_Lat=Cell_Sample_Lat,Cell_Flash_R_Occup=Cell_Flash_R_Occup,
                 Cell_duration=Cell_duration,Nb_flashes=Nb_flashes,Nb_CG=Nb_CG,T_start=T_start,T_end=T_end)

    np.savez(Path(Wdir,DAY,'ECTA',domain+'_Cells_ID_list'),CELLS_ID_list=CELLS_ID_filtered)
    print('List of cells:')
    print(CELLS_ID_filtered)

    print("END -- %s seconds ---" % (time.time() - start_time)) 
            
