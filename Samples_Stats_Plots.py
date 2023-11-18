# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:57:57 2020

@author: ronan
"""


"""
Created on Fri Oct  2 11:44:05 2020

@author: hour
"""
import glob
import os, os.path
import numpy as np
np.set_printoptions(threshold=np.inf)
import scipy
import time
from scipy import stats
import matplotlib
from matplotlib.dates import num2date
from matplotlib.ticker import MultipleLocator,AutoMinorLocator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.colors import DivergingNorm
from matplotlib.colors import LinearSegmentedColormap
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
start_time = time.time()
from pathlib import Path

#pip install matplotlib-label-lines
#from labellines import labelLine, labelLines


#PYTHON 3.7.6



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
cmap_blues=plt.get_cmap('Blues')
cmap_flag=plt.get_cmap('flag')
cmap_pastel=plt.get_cmap('tab20c')
cmap_prism=plt.get_cmap('prism')
cmap_grey=plt.get_cmap('Greys')
cmap_viridis=plt.get_cmap('viridis')
cmap_bwr=plt.get_cmap('bwr')
cmap_PRGN=plt.get_cmap('PRGn')
cmap_gist_r=plt.get_cmap('gist_rainbow_r')
cmap_coolwarm=plt.get_cmap('coolwarm')
cmap_div=plt.get_cmap('RdYlGn')

def density_points(X,Y,x_bin,y_bin):
    hist, xedges, yedges=np.histogram2d(X,Y,[x_bin,y_bin])
    xbins=0.5*(xedges[0:-1]+xedges[1::])
    ybins=0.5*(yedges[0:-1]+yedges[1::])
    return xbins,ybins,hist

def Convert_time(minutes):  
    hour, min = divmod(minutes, 60) 
    return "%d:%02d" % (hour, min) 

def Convert_time_UTC(time):
    A=matplotlib.dates.num2date(time)
    H="%02d" % A.hour+"%02d" % A.minute
    return H 

def Convert_Array_time_UTC(Array_time):
    Array=[]
    for i in Array_time:
        A=matplotlib.dates.num2date(i)
        H="%02d" % A.hour+"%02d" % A.minute
        Array=np.append(Array,H)
    return Array 



params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 15, # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'font.size': 10, # was 10
    'legend.fontsize': 14, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'text.usetex': False,
    #'figure.figsize': [19.2, 10.8],    
    'font.family': 'sans-serif',                
    #Tight layout pour optimiser l'espace entre les subplots 
    
    #Figsize * dpi= taille figure en pixels (sans tight,bbox), si besoin de précision et de zoomer prendre grosse résolution
    
    #BBox inches=coupe le blanc autour de la figure, réduit la resolution en pixels
}
matplotlib.rcParams.update(params)

#size for plot 

s_CG=100
s_VHF=3
s_IC=30
coeff=5 #multiple of s for flash plot

###############################################################
                        #File reading
###############################################################
##### ATTENTION SUPPRIME TT LES WARNINGS EN OUTPUT#######"""
H_max=15
warnings.filterwarnings("ignore")

LMA='SAETTA'

Wdir=Path('Path/To/L2b_SAETTA') #Change here the path to your work directory 

if LMA=='SAETTA':
    #Wdir=Path('/home/hour/WORK/L2b/L2b_SAETTA/')
    #Wdir=Path('E:\Ronan_work\hour\WORK\L2b\L2b_SAETTA')
    GRID_LATLON2D_CORSICA_1km=np.load(Path(Wdir/'GRID_LATLON2D_CORSICA_1km.npz'))
    LON2D=GRID_LATLON2D_CORSICA_1km['LON2D']  #to get that, i did meshgrid with XD and YD --> 2D array in meter and then m(X2D,Y2D)
    LAT2D=GRID_LATLON2D_CORSICA_1km['LAT2D']
    lat_min_REGION=41 
    lat_max_REGION=43.5
    #lon_min_REGION=6.5
    lon_min_REGION=7.5    #not to far from the center of the network 
    lon_max_REGION=10.6
    
    
domain='JJASO_SAMPLES_ESS' #Months_SAMPLES_NamePaper
List_of_months=['JUNE18','JULY18','AUGUST18','SEPTEMBER18','OCTOBER18']
DATES=['1806*','1807*','1808*','1809*','1810*']


   # Create directory
dirName = domain+'_STATS'
try:    # Create target Directory
    #os.mkdir(Wdir+'/'+dirName)
    os.mkdir(Path(Wdir/dirName))
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")


Save_path=Wdir/dirName

Z=np.arange(0,15,0.5)
DAYS_paths=[]
DOMAINS=[]
for t in range(len(List_of_months)):
    #DAYS_paths=DAYS_paths+sorted(glob.glob('/home/hour/WORK/L2b/L2b_SAETTA/'+DATES[t]))
    DAYS_paths=DAYS_paths+sorted(Wdir.glob(DATES[t]))
    #DOMAINS=np.append(DOMAINS,np.repeat(List_of_months[t],len(sorted(glob.glob('/home/hour/WORK/L2b/L2b_SAETTA/'+DATES[t])))))
    DOMAINS=np.append(DOMAINS,np.repeat(List_of_months[t],len(sorted(Wdir.glob(DATES[t])))))

DAYS=[]
for i in DAYS_paths:
    DAYS=np.append(DAYS,os.path.basename(i))


print('START STATS ON ALL CELL OF THE DAY/DAYS')
CELLS_ID_ALL=np.empty(0)
CELLS_DAY_ALL=np.empty(0)
CELLS_DOMAINS=np.empty(0)
for i in range(len(DAYS)):
    #CELLS_ID_file=np.load(Wdir+DAYS[i]+'/ECTA/'+DOMAINS[i]+'_Cells_ID_list.npz') #List of all cells we want to do stats on 
    name_Path_list=DOMAINS[i]+'_Cells_ID_list.npz'
    CELLS_ID_file=np.load(Path(Wdir/DAYS[i]/'ECTA'/name_Path_list)) #List of all cells we want to do stats on 
    CELLS_ID=CELLS_ID_file['CELLS_ID_list']
    if np.any(CELLS_ID==-999):
        continue
    CELLS_ID_ALL=np.append(CELLS_ID_ALL,CELLS_ID+int(DAYS[i])*10000)
    CELLS_DAY_ALL=np.append(CELLS_DAY_ALL,np.repeat(DAYS[i],len(CELLS_ID)))
    CELLS_DOMAINS=np.append(CELLS_DOMAINS,np.repeat(DOMAINS[i],len(CELLS_ID)))
CELLS_ID_ALL=CELLS_ID_ALL.astype(np.int64)


#cells with a delta T > 20 min (time bewteen 2 flashes)

#Cells with less than 20 flashes
Nb_cells=len(CELLS_ID_ALL)
print('# cells before filters:',Nb_cells)
Cells_Nb_flashes=np.zeros(Nb_cells)
Cells_DeltaT_max_flashes=np.zeros(len(CELLS_ID_ALL)) 

#filter process
for C in range(len(CELLS_ID_ALL)):
    Cell_id=int(CELLS_ID_ALL[C])
    print(Cell_id)
    D=CELLS_DAY_ALL[C]
    id_cell_num=int(CELLS_ID_ALL[C])-int(D)*10000
    # Load_path=Wdir+D+'/ECTA/'+CELLS_DOMAINS[C]+'_Cell_'+str(id_cell_num)
    # CELLfile = np.load(Load_path+'/CELL_'+CELLS_DOMAINS[C]+'_'+str(id_cell_num)+'.npz')
    name_load_path=CELLS_DOMAINS[C]+'_Cell_'+str(id_cell_num)
    Load_path_dir=Path(Wdir/D/'ECTA'/name_load_path)
    path_file='CELL_'+CELLS_DOMAINS[C]+'_'+str(id_cell_num)+'.npz'
    CELLfile = np.load(Path(Load_path_dir/path_file))
    Nb_flashes=CELLfile['Nb_flashes']
    Cells_Nb_flashes[C]=Nb_flashes
    Flash_time_norm=CELLfile['Flash_time_norm']
    Cell_duration=CELLfile['Cell_duration']
    T=Flash_time_norm*Cell_duration
    T_max=np.max(np.diff(T))
    Cells_DeltaT_max_flashes[C]=T_max
    
#cells with a delta T > 20 min (time bewteen 2 flashes)
CELLS_ID_ALL=CELLS_ID_ALL[Cells_DeltaT_max_flashes<20]
CELLS_DAY_ALL=CELLS_DAY_ALL[Cells_DeltaT_max_flashes<20]
CELLS_DOMAINS=CELLS_DOMAINS[Cells_DeltaT_max_flashes<20]
Cells_Nb_flashes=Cells_Nb_flashes[Cells_DeltaT_max_flashes<20]
Cells_DeltaT_max_flashes=Cells_DeltaT_max_flashes[Cells_DeltaT_max_flashes<20]
print('# cells after filter delta T flash:'+str(len(CELLS_ID_ALL))) 

#Cells with less than 20 flashes
CELLS_ID_ALL=CELLS_ID_ALL[Cells_Nb_flashes>20]
CELLS_DAY_ALL=CELLS_DAY_ALL[Cells_Nb_flashes>20]
CELLS_DOMAINS=CELLS_DOMAINS[Cells_Nb_flashes>20]
Cells_DeltaT_max_flashes=Cells_DeltaT_max_flashes[Cells_Nb_flashes>20]
Cells_Nb_flashes=Cells_Nb_flashes[Cells_Nb_flashes>20]
print('# cells after filter 20 flashes:'+str(len(CELLS_ID_ALL))) 


Nb_cells=len(CELLS_ID_ALL)

Cells_Samples_Time=np.empty(0)
Cells_Samples_Cell_Id=np.empty(0)
Cells_Samples_Nb_Flashes=np.empty(0)
Cells_Samples_Nb_Flashes_ACLR=np.empty(0)
Cells_Samples_DPL_Alt=np.empty(0)
Cells_Samples_DNL_Alt=np.empty(0)
Cells_Samples_DPL_Alt_Std=np.empty(0)
Cells_Samples_DNL_Alt_Std=np.empty(0)
Cells_Samples_Mn_SCstart=np.empty(0)
Cells_Samples_Nb_NCG=np.empty(0)
Cells_Samples_Nb_PCG=np.empty(0)
Cells_Samples_Nb_IC=np.empty(0)
Cells_Samples_Nb_NOMET=np.empty(0)
Cells_Samples_Test_Nb_Flashes=np.empty(0)
Cells_Samples_T2Cell=np.empty(0)
Cells_Samples_Nb_PIC=np.empty(0)
Cells_Samples_Nb_NIC=np.empty(0)
Cells_Samples_Nb_Dual_IC=np.empty(0)
Cells_Samples_Nb_PIC_NCG=np.empty(0)
Cells_Samples_Nb_NIC_NCG=np.empty(0)
Cells_Samples_Nb_DIC_NCG=np.empty(0)
Cells_Samples_Nb_LD=np.empty(0)
Cells_Samples_Flashes_Trigger_Alt_Median=np.empty(0)
Cells_Samples_NCG_Trigger_Alt_Median=np.empty(0)
Cells_Samples_PCG_Trigger_Alt_Median=np.empty(0)
Cells_Samples_NICHybNCG_Trigger_Alt_Median=np.empty(0)
Cells_Samples_PICHybNCG_Trigger_Alt_Median=np.empty(0)
Cells_Samples_Dual_ICHybNCG_Trigger_Alt_Median=np.empty(0)
Cells_Samples_NIC_Trigger_Alt_Median=np.empty(0)
Cells_Samples_PIC_Trigger_Alt_Median=np.empty(0)
Cells_Samples_Dual_IC_Trigger_Alt_Median=np.empty(0)
Cells_Samples_NOMET_Trigger_Alt_Median=np.empty(0)
Cells_Samples_NOMET_V_Extent_Median=np.empty(0)
Cells_Samples_H_Extent=np.empty(0)
Cells_Samples_NCG_Multistrokes=np.empty(0)
Cells_Samples_Nb_Long_Flashes=np.empty(0)
Cells_Samples_ID=np.empty(0)
Cells_Samples_Lat=np.empty(0)
Cells_Samples_Lon=np.empty(0)
Cells_Samples_Hour=np.empty(0)
Cells_Samples_R_Occup_Median=np.empty(0)
Cells_Samples_Nb_Potential_DPL=np.empty(0)
Cells_Samples_Nb_Potential_DNL=np.empty(0)
Cells_Samples_Bad_DNL_Sample=np.empty(0)
Cells_Samples_Bad_DPL_Sample=np.empty(0)
Cells_Samples_NCG_Multiplicity=np.empty(0)
All_Flashes_Duration=np.empty(0)
All_Flashes_Trigger_Alt=np.empty(0)
All_Flashes_Class_Main=np.empty(0)
All_Flashes_Class_Complete=np.empty(0)
All_Flashes_Current=np.empty(0)
All_Flashes_Samples_ID=np.empty(0)
All_Flashes_Flag_f_ACLR=np.empty(0)
All_Flashes_R_Occup=np.empty(0)
All_Flashes_Nb_S=np.empty(0)
All_Flashes_Lon=np.empty(0)
All_Flashes_Lat=np.empty(0)
All_Flashes_1st_S_Flag=np.empty(0)
Cells_Duration=np.empty(0)
All_MET_Current=np.empty(0)
All_MET_Time=np.empty(0)
All_MET_Type=np.empty(0)
Cells_NB_PCG=np.empty(0)
Cells_NB_NCG=np.empty(0)

ID=0
for C in range(len(CELLS_ID_ALL)):
#for C in [0]:
    Cell_id=int(CELLS_ID_ALL[C])
    print(Cell_id)
    D=CELLS_DAY_ALL[C]
    id_cell_num=int(CELLS_ID_ALL[C])-int(D)*10000
    # Load_path=Wdir+D+'/ECTA/'+CELLS_DOMAINS[C]+'_Cell_'+str(id_cell_num)
    # CELLfile = np.load(Load_path+'/CELL_'+CELLS_DOMAINS[C]+'_'+str(id_cell_num)+'.npz')
    name_load_path=CELLS_DOMAINS[C]+'_Cell_'+str(id_cell_num)
    Load_path_dir=Path(Wdir/D/'ECTA'/name_load_path)
    path_file='CELL_'+CELLS_DOMAINS[C]+'_'+str(id_cell_num)+'.npz'
    CELLfile = np.load(Path(Load_path_dir/path_file))
    Cell_duration=CELLfile['Cell_duration']
    T_start=CELLfile['T_start']
    T_end=CELLfile['T_end']
    Flash_time_norm=CELLfile['Flash_time_norm']
    Flash_class_main=CELLfile['Cell_Flash_class_main']
    Flash_class_complete=CELLfile['Cell_Flash_class_complete']
    Flash_trigger_alt=CELLfile['Flash_trigger_alt'] 
    L_H=CELLfile['L_H']
    Flash_vertical_extension=CELLfile['Flash_vertical_extension']
    Multiplicity=CELLfile['Multiplicity']
    F_Duration=CELLfile['Duration']
    F_Nb_S=CELLfile['Nb_S']
    Flag_f_ACLR=CELLfile['Flag_f_ACLR']
    Cell_LMA_lon=CELLfile['Cell_LMA_lon']
    Cell_LMA_lat=CELLfile['Cell_LMA_lat']
    Flash_1st_S_Flag=CELLfile['Flash_first_source_flag']
    
    Cell_MET_Current=CELLfile['Cell_MET_current']
    Cell_MET_Time=CELLfile['Cell_MET_strk_to_LMA']
    Cell_MET_Type=CELLfile['Cell_MET_type']
    
    Cells_Duration=np.append(Cells_Duration,Cell_duration)
    Cells_NB_NCG=np.append(Cells_NB_NCG,len(Flash_class_main[Flash_class_main=='-CG']))
    Cells_NB_PCG=np.append(Cells_NB_PCG,len(Flash_class_main[Flash_class_main=='+CG']))
    
    Samples_Time=CELLfile['Cell_time_interval']
    Cells_Samples_Time=np.append(Cells_Samples_Time,Samples_Time)
    Cell_trajectory_2D=CELLfile['Cell_trajectory_2D']
    
    
    Samples_Cell_Id=np.repeat(Cell_id,len(Samples_Time))
    Cells_Samples_Cell_Id=np.append(Cells_Samples_Cell_Id,Samples_Cell_Id)
    
    Samples_Nb_F=CELLfile['Cell_Nb_F_interval']
    Cells_Samples_Nb_Flashes=np.append(Cells_Samples_Nb_Flashes,Samples_Nb_F)
   
    Samples_Nb_F_ACLR=CELLfile['Cell_Nb_F_ACLR_interval']
    Cells_Samples_Nb_Flashes_ACLR=np.append(Cells_Samples_Nb_Flashes_ACLR,Samples_Nb_F_ACLR)
    
    Samples_PL_max=CELLfile['Cell_PL_interval_max']
    Samples_NL_max=CELLfile['Cell_NL_interval_max']
    
    Cell_PL_Sample=CELLfile['Cell_PL_Sample']
    Cell_NL_Sample=CELLfile['Cell_NL_Sample']   #PEUT SERVIR 
    
    Samples_Lon=CELLfile['Cell_Sample_Lon']
    Samples_Lat=CELLfile['Cell_Sample_Lat']
    
    
    Flashes_R_Occup=CELLfile['Cell_Flash_R_Occup']
    

    Samples_T2Cell=(Samples_Time-T_start)/(T_end-T_start)
    Cells_Samples_T2Cell=np.append(Cells_Samples_T2Cell,Samples_T2Cell)
    
    Samples_DPL_Alt=np.zeros(len(Samples_Time))
    Samples_DNL_Alt=np.zeros(len(Samples_Time))
    Sample_Nb_Potential_DPL=np.zeros(len(Samples_Time))
    Sample_Nb_Potential_DNL=np.zeros(len(Samples_Time))
    Samples_Nb_F_4_DPL=np.zeros(len(Samples_Time))
    Samples_Nb_F_4_DNL=np.zeros(len(Samples_Time))
    Bad_DNL_Sample=np.zeros(len(Samples_Time))
    Bad_DPL_Sample=np.zeros(len(Samples_Time))
    Sample_DNL_Alt_Std=np.zeros(len(Samples_Time))
    Sample_DPL_Alt_Std=np.zeros(len(Samples_Time))
    for i in range(0,len(Samples_Time),1):
        Sample_Nb_Potential_DPL[i]=len(Samples_PL_max[i,:][Samples_PL_max[i,:]==1])
        Sample_Nb_Potential_DNL[i]=len(Samples_NL_max[i,:][Samples_NL_max[i,:]==1])
        Samples_Nb_F_4_DPL[i]=np.max(Cell_PL_Sample[i,:])
        Samples_Nb_F_4_DNL[i]=np.max(Cell_NL_Sample[i,:])
        Samples_DPL_Alt[i]=np.mean(Z[Samples_PL_max[i,:]==np.max(Samples_PL_max[i,:])])
        Samples_DNL_Alt[i]=np.mean(Z[Samples_NL_max[i,:]==np.max(Samples_NL_max[i,:])])
        Sample_DNL_Alt_Std[i]=np.std(Z[Samples_NL_max[i,:]==np.max(Samples_NL_max[i,:])])
        Sample_DPL_Alt_Std[i]=np.std(Z[Samples_PL_max[i,:]==np.max(Samples_PL_max[i,:])])
        if len(Cell_NL_Sample[i,:][Cell_NL_Sample[i,:]>0])==0:
            Bad_DNL_Sample[i]=1
        if len(Cell_PL_Sample[i,:][Cell_PL_Sample[i,:]>0])==0:
            Bad_DPL_Sample[i]=1              
            
    
    Cells_Samples_DPL_Alt=np.append(Cells_Samples_DPL_Alt,Samples_DPL_Alt)
    Cells_Samples_DNL_Alt=np.append(Cells_Samples_DNL_Alt,Samples_DNL_Alt)
    Cells_Samples_DNL_Alt_Std=np.append(Cells_Samples_DNL_Alt_Std,Sample_DNL_Alt_Std)
    Cells_Samples_DPL_Alt_Std=np.append(Cells_Samples_DPL_Alt_Std,Sample_DPL_Alt_Std)
    #Flash_Min_SCstart=Flash_time_norm*Cell_duration
    Flash_Trigger_Time=(T_end-T_start)*Flash_time_norm+T_start
    Samples_Mn_SCstart=np.arange(10,len(Samples_Time)*10+10,10)  #always end of the 10 min period 
    Cells_Samples_Mn_SCstart=np.append(Cells_Samples_Mn_SCstart,Samples_Mn_SCstart)
    
    #from 0 to 1 for the cell life
    
    Samples_Nb_CG=np.zeros(len(Samples_Time))
    Samples_Nb_NCG=np.zeros(len(Samples_Time))
    Samples_Nb_PCG=np.zeros(len(Samples_Time))
    Samples_Nb_IC=np.zeros(len(Samples_Time))
    Samples_Nb_PIC=np.zeros(len(Samples_Time))
    Samples_Nb_NIC=np.zeros(len(Samples_Time))
    Samples_Nb_Dual_IC=np.zeros(len(Samples_Time))
    Samples_Nb_NOMET=np.zeros(len(Samples_Time))
    Samples_Hour=np.zeros(len(Samples_Time))
    Test_Nb_Flashes=np.zeros(len(Samples_Time))
    Samples_Nb_PIC_NCG=np.zeros(len(Samples_Time))
    Samples_Nb_NIC_NCG=np.zeros(len(Samples_Time))
    Samples_Nb_DIC_NCG=np.zeros(len(Samples_Time))
    Samples_Nb_Long_Flashes=np.zeros(len(Samples_Time))
    Samples_Flashes_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_NCG_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_PCG_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_NICHybNCG_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_PICHybNCG_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_Dual_ICHybNCG_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_NIC_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_PIC_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_Dual_IC_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_NOMET_Trigger_Alt_Median=np.zeros(len(Samples_Time))
    Samples_NOMET_V_Extent_Median=np.zeros(len(Samples_Time))
    Samples_NCG_Multistrokes=np.zeros(len(Samples_Time))
    Samples_R_Occup_Median=np.zeros(len(Samples_Time))
    #Samples_H_Extent=np.zeros(len(Samples_Time))
    Samples_DPL_Alt_TEMP=np.zeros(len(Samples_Time))
    Samples_DNL_Alt_TEMP=np.zeros(len(Samples_Time))
    Samples_Mass_Water=np.zeros(len(Samples_Time))
    Samples_Nb_LD=np.zeros(len(Samples_Time)) #flash in lower dipole (under a certain inititaion height) --> under 6 km
    

    F=CELLfile['F']
    FED_EXTRACT_TIME=CELLfile['FED_EXTRACT_TIME']
    FED_5min=CELLfile['FED_5min']
    FED_5min=np.ma.masked_where(FED_5min<=0,FED_5min)
    Cell_LMA_time=CELLfile['Cell_LMA_time']
    Cell_LMA_flash_class_main=CELLfile['Cell_LMA_flash_class_main']    
    Cell_LMA_flash=CELLfile['Cell_LMA_flash']
    Cell_5min_FRT=np.zeros(len(FED_EXTRACT_TIME))
    Cell_5min_FRCG=np.zeros(len(FED_EXTRACT_TIME))
    Cell_5min_FRIC=np.zeros(len(FED_EXTRACT_TIME))
    Cell_5min_FRLL=np.zeros(len(FED_EXTRACT_TIME))  #LOW flash (trigger under 4.5 km )
    Cell_5min_Trig_Alt_CG=np.zeros(len(FED_EXTRACT_TIME))
    Cell_5min_Trig_Alt_TL=np.zeros(len(FED_EXTRACT_TIME))
    Cell_5min_Trig_Alt_IC=np.zeros(len(FED_EXTRACT_TIME))
    Cell_5min_Trig_Alt_LL=np.zeros(len(FED_EXTRACT_TIME))
    for i in range(len(FED_EXTRACT_TIME)):
        Time_condition=(Cell_LMA_time<=FED_EXTRACT_TIME[i]) & (Cell_LMA_time>FED_EXTRACT_TIME[i]-5/24./60.)
        Time_condition_F=(Flash_Trigger_Time<=FED_EXTRACT_TIME[i]) & (Flash_Trigger_Time>FED_EXTRACT_TIME[i]-5/24./60.)
        Cell_5min_FRT[i]=len(np.unique(Cell_LMA_flash[Time_condition]))  ### Possible double count of flashes between 2 periods....  that important each 5 min ?
        Cell_5min_FRCG[i]=len(np.unique(Cell_LMA_flash[(Time_condition) & ((Cell_LMA_flash_class_main=='+CG') |(Cell_LMA_flash_class_main=='-CG'))]))
        Cell_5min_FRIC[i]=len(np.unique(Cell_LMA_flash[(Time_condition) & (Cell_LMA_flash_class_main!='+CG') & (Cell_LMA_flash_class_main!='-CG')]))
        Cell_5min_FRLL[i]=len(np.unique(Cell_LMA_flash[(Time_condition) & np.isin(Cell_LMA_flash,F[Flash_trigger_alt<4.5])]))
        Cell_5min_Trig_Alt_CG[i]=np.mean(Flash_trigger_alt[Time_condition_F & ((Flash_class_main=='+CG') |(Flash_class_main=='-CG'))])
        Cell_5min_Trig_Alt_IC[i]=np.mean(Flash_trigger_alt[Time_condition_F & ((Flash_class_main!='+CG') & (Flash_class_main!='-CG'))])
        Cell_5min_Trig_Alt_TL[i]=np.mean(Flash_trigger_alt[Time_condition_F])
        Cell_5min_Trig_Alt_LL[i]=np.mean(Flash_trigger_alt[(Time_condition_F) & (Flash_trigger_alt<4.5)])
        
    for T in range(0,len(Samples_Time),1): 
        #print(Convert_time_UTC(Cell_time_interval[i]))
        if T==0:
            F_Time_condition=Flash_Trigger_Time<=Samples_Time[T]
            MET_Time_condition=Cell_MET_Time<=Samples_Time[T]

        else:
            F_Time_condition=(Flash_Trigger_Time<=Samples_Time[T]) & (Flash_Trigger_Time>Samples_Time[T-1])
            MET_Time_condition=(Cell_MET_Time<=Samples_Time[T]) & (Cell_MET_Time>Samples_Time[T-1])
           
            
        Test_Nb_Flashes[T]=len(Flash_time_norm[F_Time_condition])
        Samples_Nb_NCG[T]=len(Flash_time_norm[(Flash_class_main=='-CG') & (F_Time_condition)])
        Samples_Nb_PCG[T]=len(Flash_time_norm[(Flash_class_main=='+CG') & (F_Time_condition)])
        Samples_Nb_IC[T]=len(Flash_time_norm[(Flash_class_main=='IC') & (F_Time_condition)])
        Samples_Nb_NOMET[T]=len(Flash_time_norm[(Flash_class_main=='NO MET') & (F_Time_condition)])
        Samples_Nb_PIC[T]=len(Flash_time_norm[(Flash_class_complete=='+IC') & (F_Time_condition)])
        Samples_Nb_NIC[T]=len(Flash_time_norm[(Flash_class_complete=='-IC') & (F_Time_condition)])
        Samples_Nb_Dual_IC[T]=len(Flash_time_norm[(Flash_class_complete=='Dual_IC') & (F_Time_condition)])
        Samples_Nb_PIC_NCG[T]=len(Flash_time_norm[(Flash_class_complete=='+IC_Hybrid_-CG') & (F_Time_condition)])
        Samples_Nb_NIC_NCG[T]=len(Flash_time_norm[(Flash_class_complete=='-IC_Hybrid_-CG') & (F_Time_condition)])
        Samples_Nb_DIC_NCG[T]=len(Flash_time_norm[(Flash_class_complete=='Dual_IC_Hybrid_-CG') & (F_Time_condition)])
        Samples_Nb_LD[T]=len(Flash_time_norm[Flash_trigger_alt<6])  #all falshes under 6 km 
        Samples_Flashes_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[F_Time_condition])
        Samples_NCG_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_main=='-CG') & (F_Time_condition)])
        Samples_PCG_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_main=='+CG') & (F_Time_condition)])
        Samples_NICHybNCG_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_complete=='-IC_Hybrid_-CG') & (F_Time_condition)])
        Samples_PICHybNCG_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_complete=='+IC_Hybrid_-CG') & (F_Time_condition)])
        Samples_Dual_ICHybNCG_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_complete=='Dual_IC_Hybrid_-CG') & (F_Time_condition)])
        Samples_PIC_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_complete=='+IC') & (F_Time_condition)])
        Samples_NIC_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_complete=='-IC') & (F_Time_condition)])
        Samples_Dual_IC_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_complete=='Dual_IC') & (F_Time_condition)])
        Samples_NOMET_Trigger_Alt_Median[T]=np.median(Flash_trigger_alt[(Flash_class_main=='NO MET') & (F_Time_condition)])
        Samples_NOMET_V_Extent_Median[T]=np.median(Flash_vertical_extension[(Flash_class_main=='NO MET') & (F_Time_condition)])
        Samples_R_Occup_Median[T]=np.median(Flashes_R_Occup[F_Time_condition])
        #print(Convert_time_UTC(Samples_Time[T]),Convert_Array_time_UTC(Cell_ACRR_Time[Mass_water_condition]),Samples_Mass_Water[T])
        Samples_Hour[T]=num2date(Samples_Time[T]).hour
        Samples_Nb_Long_Flashes[T]=len(F_Duration[(F_Duration>0.1) & (F_Time_condition)])  #Nb of flashes witha  duration >0.1 s 
        if np.any(Multiplicity[(Flash_class_main=='-CG') & (F_Time_condition)])==True:
            Samples_NCG_Multistrokes[T]=len(Multiplicity[(Flash_class_main=='-CG') & (F_Time_condition) & (Multiplicity>1)])/len(Flash_time_norm[(Flash_class_main=='-CG') & (F_Time_condition)]) # -CGs with multiple strokes/-CGs
        else:
            Samples_NCG_Multistrokes[T]=np.nan
        
        All_Flashes_Duration=np.append(All_Flashes_Duration,F_Duration[F_Time_condition])
        All_Flashes_Trigger_Alt=np.append(All_Flashes_Trigger_Alt,Flash_trigger_alt[F_Time_condition])
        All_Flashes_Class_Main=np.append(All_Flashes_Class_Main,Flash_class_main[F_Time_condition])
        All_Flashes_Class_Complete=np.append(All_Flashes_Class_Complete,Flash_class_complete[F_Time_condition])
        All_Flashes_Samples_ID=np.append(All_Flashes_Samples_ID,np.repeat(ID,len(Flash_time_norm[F_Time_condition])))
        All_Flashes_Flag_f_ACLR=np.append(All_Flashes_Flag_f_ACLR,Flag_f_ACLR[F_Time_condition])
        All_Flashes_R_Occup=np.append(All_Flashes_R_Occup,Flashes_R_Occup[F_Time_condition])
        All_Flashes_Nb_S=np.append(All_Flashes_Nb_S,F_Nb_S[F_Time_condition])
        All_Flashes_Lat=np.append(All_Flashes_Lat,Cell_LMA_lat[Flash_1st_S_Flag==1][F_Time_condition])
        All_Flashes_Lon=np.append(All_Flashes_Lon,Cell_LMA_lon[Flash_1st_S_Flag==1][F_Time_condition])
        
        All_MET_Current=np.append(All_MET_Current,Cell_MET_Current[MET_Time_condition])
        All_MET_Type=np.append(All_MET_Type,Cell_MET_Type[MET_Time_condition])
        All_MET_Time=np.append(All_MET_Time,Cell_MET_Time[MET_Time_condition])
        
        Cells_Samples_ID=np.append(Cells_Samples_ID,ID)
        
        
        
        ID=ID+1
        
  
    Cells_Samples_Nb_NCG=np.append(Cells_Samples_Nb_NCG, Samples_Nb_NCG)
    Cells_Samples_Nb_PCG=np.append(Cells_Samples_Nb_PCG,Samples_Nb_PCG)
    Cells_Samples_Nb_IC=np.append(Cells_Samples_Nb_IC,Samples_Nb_IC)
    Cells_Samples_Nb_NOMET=np.append(Cells_Samples_Nb_NOMET,Samples_Nb_NOMET)
    Cells_Samples_Test_Nb_Flashes=np.append( Cells_Samples_Test_Nb_Flashes,Test_Nb_Flashes)
    Cells_Samples_Nb_PIC=np.append(Cells_Samples_Nb_PIC,Samples_Nb_PIC)
    Cells_Samples_Nb_NIC=np.append(Cells_Samples_Nb_NIC,Samples_Nb_NIC)
    Cells_Samples_Nb_Dual_IC=np.append(Cells_Samples_Nb_Dual_IC,Samples_Nb_Dual_IC)
    Cells_Samples_Nb_PIC_NCG=np.append(Cells_Samples_Nb_PIC_NCG,Samples_Nb_PIC_NCG)
    Cells_Samples_Nb_NIC_NCG=np.append(Cells_Samples_Nb_NIC_NCG, Samples_Nb_NIC_NCG)
    Cells_Samples_Nb_DIC_NCG=np.append(Cells_Samples_Nb_DIC_NCG, Samples_Nb_DIC_NCG)
    Cells_Samples_Nb_Long_Flashes=np.append(Cells_Samples_Nb_Long_Flashes,Samples_Nb_Long_Flashes)
    Cells_Samples_Nb_LD=np.append(Cells_Samples_Nb_LD,Samples_Nb_LD)
    Cells_Samples_Flashes_Trigger_Alt_Median=np.append(Cells_Samples_Flashes_Trigger_Alt_Median,Samples_Flashes_Trigger_Alt_Median)
    Cells_Samples_NCG_Trigger_Alt_Median=np.append(Cells_Samples_NCG_Trigger_Alt_Median,Samples_NCG_Trigger_Alt_Median)
    Cells_Samples_PCG_Trigger_Alt_Median=np.append(Cells_Samples_PCG_Trigger_Alt_Median,Samples_PCG_Trigger_Alt_Median)
    Cells_Samples_NICHybNCG_Trigger_Alt_Median=np.append(Cells_Samples_NICHybNCG_Trigger_Alt_Median,Samples_NICHybNCG_Trigger_Alt_Median)
    Cells_Samples_PICHybNCG_Trigger_Alt_Median=np.append(Cells_Samples_PICHybNCG_Trigger_Alt_Median,Samples_PICHybNCG_Trigger_Alt_Median)
    Cells_Samples_Dual_ICHybNCG_Trigger_Alt_Median=np.append(Cells_Samples_Dual_ICHybNCG_Trigger_Alt_Median,Samples_Dual_ICHybNCG_Trigger_Alt_Median)
    Cells_Samples_Dual_IC_Trigger_Alt_Median=np.append(Cells_Samples_Dual_IC_Trigger_Alt_Median,Samples_Dual_IC_Trigger_Alt_Median)
    Cells_Samples_PIC_Trigger_Alt_Median=np.append(Cells_Samples_PIC_Trigger_Alt_Median,Samples_PIC_Trigger_Alt_Median)
    Cells_Samples_NIC_Trigger_Alt_Median=np.append(Cells_Samples_NIC_Trigger_Alt_Median,Samples_NIC_Trigger_Alt_Median)
    Cells_Samples_NOMET_Trigger_Alt_Median=np.append(Cells_Samples_NOMET_Trigger_Alt_Median,Samples_NOMET_Trigger_Alt_Median)
    Cells_Samples_Lon=np.append(Cells_Samples_Lon,Samples_Lon)
    Cells_Samples_Lat=np.append(Cells_Samples_Lat,Samples_Lat)
    Cells_Samples_Hour=np.append(Cells_Samples_Hour,Samples_Hour)
    Cells_Samples_NOMET_V_Extent_Median=np.append(Cells_Samples_NOMET_V_Extent_Median,Samples_NOMET_V_Extent_Median)
    Cells_Samples_NCG_Multistrokes=np.append(Cells_Samples_NCG_Multistrokes,Samples_NCG_Multistrokes)
    Cells_Samples_R_Occup_Median=np.append(Cells_Samples_R_Occup_Median,Samples_R_Occup_Median)
    Cells_Samples_Nb_Potential_DPL=np.append(Cells_Samples_Nb_Potential_DPL,Sample_Nb_Potential_DPL)
    Cells_Samples_Nb_Potential_DNL=np.append(Cells_Samples_Nb_Potential_DNL,Sample_Nb_Potential_DNL)
    Cells_Samples_Bad_DNL_Sample=np.append(Cells_Samples_Bad_DNL_Sample,Bad_DNL_Sample)
    Cells_Samples_Bad_DPL_Sample=np.append(Cells_Samples_Bad_DPL_Sample,Bad_DPL_Sample)
         


GEO_CONDITION=Cells_Samples_Lon>=lon_min_REGION


Cells_Samples_Nb_Flashes=Cells_Samples_Nb_Flashes.astype(np.int64)
Cells_Samples_Nb_Flashes_ACLR= Cells_Samples_Nb_Flashes_ACLR.astype(np.int64)


Cells_Samples_Nb_Flashes_GEO=Cells_Samples_Nb_Flashes[GEO_CONDITION].astype(np.int64)
Cells_Samples_Nb_Flashes_ACLR_GEO= Cells_Samples_Nb_Flashes_ACLR[GEO_CONDITION].astype(np.int64)



Cells_Samples_Conf_R=Cells_Samples_Nb_Flashes_ACLR/Cells_Samples_Nb_Flashes*100
Cells_Samples_Conf_R_GEO=Cells_Samples_Nb_Flashes_ACLR_GEO/Cells_Samples_Nb_Flashes_GEO*100

Cells_Samples_Conf_R[np.isnan(Cells_Samples_Conf_R)]=0  #When no flashes, gives nan --> 0
Cells_Samples_Conf_R_GEO[np.isnan(Cells_Samples_Conf_R_GEO)]=0  #When no flashes, gives nan --> 0


Cells_Samples_Polarity=np.zeros(len(Cells_Samples_DNL_Alt))
Cells_Samples_Polarity[Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt]=1
Cells_Samples_Polarity[Cells_Samples_DNL_Alt>Cells_Samples_DPL_Alt]=-1

Cells_Samples_Polarity_GEO=np.zeros(len(Cells_Samples_DNL_Alt[GEO_CONDITION]))
Cells_Samples_Polarity_GEO[Cells_Samples_DPL_Alt[GEO_CONDITION]>Cells_Samples_DNL_Alt[GEO_CONDITION]]=1
Cells_Samples_Polarity_GEO[Cells_Samples_DNL_Alt[GEO_CONDITION]>Cells_Samples_DPL_Alt[GEO_CONDITION]]=-1

  

Cells_Samples_Cell_Id_GEO=Cells_Samples_Cell_Id[GEO_CONDITION]

print('# CELLS AFTER GEO CONDITION:',len(np.unique(Cells_Samples_Cell_Id_GEO)))
########################## CELLS TRAJECTORIES ##################
print('Cells trajectories plot:')
fig=plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111,projection=ccrs.Mercator())   #plot map
ax.set_extent([lon_min_REGION,lon_max_REGION,lat_min_REGION,lat_max_REGION])
ax.add_feature(cfeature.STATES.with_scale('10m'),linestyle='--', alpha=.5)
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND.with_scale('10m'))
ax.add_feature(cfeature.OCEAN.with_scale('10m'))
ax.add_feature(cfeature.STATES.with_scale('10m'),linestyle='--', alpha=.5)


#gl=ax.gridlines(draw_labels=True,linestyle='--',color='gray',alpha=0.5)
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '10m',facecolor=cfeature.COLORS['water']))
# ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m',facecolor=cfeature.COLORS['land']))

Cells_Samples_Lat_GEO=Cells_Samples_Lat[GEO_CONDITION]
Cells_Samples_Lon_GEO=Cells_Samples_Lon[GEO_CONDITION]
#gl.xlabels_top=False
#gl.ylabels_right=False
#gl.xformatter = LONGITUDE_FORMATTER
#gl.yformatter = LATITUDE_FORMATTER 
for i in CELLS_ID_ALL:
    condition=(Cells_Samples_Cell_Id_GEO==i)
    if i==1807260002:
        ax.plot(Cells_Samples_Lon_GEO[condition],Cells_Samples_Lat_GEO[condition],color='red',alpha=1,transform=ccrs.PlateCarree(),zorder=2.5)
        ax.scatter(Cells_Samples_Lon_GEO[condition][-1],Cells_Samples_Lat_GEO[condition][-1],marker="x",color='red',alpha=1,s=25,transform=ccrs.PlateCarree(),zorder=2)
    if np.isin(i,Cells_Samples_Cell_Id_GEO)==True:
        #ax.plot(Cells_Samples_Lon[Cells_Samples_Cell_Id==i],Cells_Samples_Lat[Cells_Samples_Cell_Id==i],color='black',alpha=0.5,transform=ccrs.PlateCarree(),zorder=2.5)
        ax.plot(Cells_Samples_Lon_GEO[condition & (~np.isnan(Cells_Samples_Lon_GEO))],Cells_Samples_Lat_GEO[(Cells_Samples_Cell_Id_GEO==i) & (~np.isnan(Cells_Samples_Lat_GEO))],transform=ccrs.PlateCarree(),zorder=2,alpha=0.5,color='black')
        ax.scatter(Cells_Samples_Lon_GEO[(Cells_Samples_Cell_Id_GEO==i) & (~np.isnan(Cells_Samples_Lon_GEO))][-1],Cells_Samples_Lat_GEO[(Cells_Samples_Cell_Id_GEO==i) & (~np.isnan(Cells_Samples_Lat_GEO))][-1],marker="x",color='black',alpha=1,s=15,transform=ccrs.PlateCarree(),zorder=2)
        if (i>1810290000) & (i<1810300000):
            ax.plot(Cells_Samples_Lon_GEO[condition],Cells_Samples_Lat_GEO[condition],color='red',alpha=1,transform=ccrs.PlateCarree(),zorder=2.5)
            ax.scatter(Cells_Samples_Lon_GEO[condition][-1],Cells_Samples_Lat_GEO[condition][-1],marker="x",color='red',alpha=1,s=25,transform=ccrs.PlateCarree(),zorder=2)

plt.tight_layout()
NAME_PLOT='Domain_Cells_Trajectories_'+domain+'.jpeg'
plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight") 
plt.close() 




print("NB CELLS: ", len(Cells_Duration))
####################
NB_SAMPLES_before=len(Cells_Samples_Time)

All_Flashes_Samples_ID
print('RAW')

 
print('#total flashes:',len(All_Flashes_Lat))
print('#ACLR flashes:',len(All_Flashes_Lat[All_Flashes_Flag_f_ACLR==1]))
print('#ACLR flashes/#Total flashes: '+str(np.round(len(All_Flashes_Lat[All_Flashes_Flag_f_ACLR==1])/len(All_Flashes_Lat)*100,2))+' %')

unique, index, inverse, counts = np.unique(All_Flashes_Samples_ID,return_index=True,return_inverse=True, return_counts=True)

Condition_Flashes=np.isin(All_Flashes_Samples_ID,Cells_Samples_ID[GEO_CONDITION])

print("############### GEO Flashes and samples ####################### ")
print('Total # Samples before GEO:',NB_SAMPLES_before)
print('Total # Samples AFTER GEO:',len(Cells_Samples_Time[GEO_CONDITION]))

print('#total flashes AFTER GEO:',len(All_Flashes_Lat[Condition_Flashes]))
print('#ACLR flashes AFTER GEO:',len(All_Flashes_Lat[(All_Flashes_Flag_f_ACLR==1) & (Condition_Flashes)]))
print('#ACLR flashes/#Total flashes AFTER GEO: '+str(np.round(len(All_Flashes_Lat[(All_Flashes_Flag_f_ACLR==1) & (Condition_Flashes)])/len(All_Flashes_Lat)*100,2))+' %')




#CELLS month association
Cells_Month=np.copy(CELLS_ID_ALL)
Cells_Month[(CELLS_ID_ALL>=1806000000) & (CELLS_ID_ALL<1807000000)]=6
Cells_Month[(CELLS_ID_ALL>=1807000000) & (CELLS_ID_ALL<1808000000)]=7
Cells_Month[(CELLS_ID_ALL>=1808000000) & (CELLS_ID_ALL<1809000000)]=8
Cells_Month[(CELLS_ID_ALL>=1809000000) & (CELLS_ID_ALL<1810000000)]=9
Cells_Month[(CELLS_ID_ALL>=1810000000) & (CELLS_ID_ALL<1811000000)]=10

A1=np.unique(Cells_Samples_Cell_Id,return_counts=True)
#print(Cells_Duration/A1[1])



##########################  FILTERS APPLICATION ############################# 
print('#flashes before filter:',len(All_Flashes_Class_Main)) 
print('#flashes before filter>20 sources:',len(All_Flashes_Class_Main[All_Flashes_Nb_S>20]))
print('#flashes used by chargepol before filter:',len(All_Flashes_Class_Main[All_Flashes_Flag_f_ACLR==1]))
   
print('Total # Samples:',NB_SAMPLES_before)
NB_SAMPLES=len(Cells_Samples_Time)
NB_NP=len(Cells_Samples_Polarity[Cells_Samples_Polarity==1]) #normal
NB_AP=len(Cells_Samples_Polarity[Cells_Samples_Polarity==-1])   #anomalous
NB_UNK=len(Cells_Samples_Polarity[Cells_Samples_Polarity==0])   #Unknown
print("SAMPLES:")
print(NB_NP,str(np.round(NB_NP/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Normal polarity')
print(NB_AP,str(np.round(NB_AP/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Anomalous polarity')


print("################# GEO ########################")
print('Total # Samples after GEO:',len(Cells_Samples_Time[GEO_CONDITION]))
NB_SAMPLES=len(Cells_Samples_Time[GEO_CONDITION])
NB_NP=len(Cells_Samples_Polarity[GEO_CONDITION][Cells_Samples_Polarity[GEO_CONDITION]==1]) #normal
NB_AP=len(Cells_Samples_Polarity[GEO_CONDITION][Cells_Samples_Polarity[GEO_CONDITION]==-1])   #anomalous
NB_UNK=len(Cells_Samples_Polarity[GEO_CONDITION][Cells_Samples_Polarity[GEO_CONDITION]==0])   #Unknown
print("SAMPLES GEO:")
print(NB_NP,str(np.round(NB_NP/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Normal polarity - GEO')
print(NB_AP,str(np.round(NB_AP/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Anomalous polarity - GEO')
print(NB_UNK,str(np.round(NB_UNK/NB_SAMPLES,2)*100)+' % of 10-min samples associated to Unknown polarity - GEO')




print('###FILTER on samples with less conf <=20%########################')

unique1, counts1 = np.unique(Cells_Samples_Cell_Id, return_counts=True)

#Condition=[Cells_Samples_Nb_Flashes_ACLR>=5]
Condition_Samples=(Cells_Samples_Conf_R>=20)
###Filter samples with Layers condition
###filter sample with no layers found (and rare case with no negative but positive)
###when std between multiple dominant bin it s too important (>2 km)
###when DPL alt==DNL alt 
Condition_Layers=(Cells_Samples_Bad_DNL_Sample==0) & (Cells_Samples_Bad_DPL_Sample==0) & (Cells_Samples_DNL_Alt_Std<=2) & (Cells_Samples_DPL_Alt_Std<=2) & (Cells_Samples_DPL_Alt!=Cells_Samples_DNL_Alt)

Condition_Global=Condition_Samples & Condition_Layers & GEO_CONDITION

Cells_Samples_Time=Cells_Samples_Time[Condition_Global]
Cells_Samples_Cell_Id=Cells_Samples_Cell_Id[Condition_Global]
Cells_Samples_Nb_Flashes=Cells_Samples_Nb_Flashes[Condition_Global]
Cells_Samples_DPL_Alt= Cells_Samples_DPL_Alt[Condition_Global]
Cells_Samples_DNL_Alt= Cells_Samples_DNL_Alt[Condition_Global]
Cells_Samples_Mn_SCstart= Cells_Samples_Mn_SCstart[Condition_Global]
Cells_Samples_Nb_NCG=Cells_Samples_Nb_NCG[Condition_Global]
Cells_Samples_Nb_PCG=Cells_Samples_Nb_PCG [Condition_Global]
Cells_Samples_Nb_IC=Cells_Samples_Nb_IC[Condition_Global]
Cells_Samples_Nb_Dual_IC=Cells_Samples_Nb_Dual_IC[Condition_Global]
Cells_Samples_Nb_PIC_NCG=Cells_Samples_Nb_PIC_NCG [Condition_Global]
Cells_Samples_Nb_NIC_NCG=Cells_Samples_Nb_NIC_NCG [Condition_Global]
Cells_Samples_Nb_DIC_NCG=Cells_Samples_Nb_DIC_NCG [Condition_Global]
Cells_Samples_Nb_LD=Cells_Samples_Nb_LD[Condition_Global]
Cells_Samples_Nb_NOMET=Cells_Samples_Nb_NOMET[Condition_Global]
Cells_Samples_Test_Nb_Flashes=Cells_Samples_Test_Nb_Flashes[Condition_Global]
Cells_Samples_T2Cell=Cells_Samples_T2Cell[Condition_Global]
Cells_Samples_Nb_NIC=Cells_Samples_Nb_NIC[Condition_Global]
Cells_Samples_Nb_PIC=Cells_Samples_Nb_PIC[Condition_Global]
Cells_Samples_Flashes_Trigger_Alt_Median=Cells_Samples_Flashes_Trigger_Alt_Median[Condition_Global]
Cells_Samples_NCG_Trigger_Alt_Median=Cells_Samples_NCG_Trigger_Alt_Median[Condition_Global]
Cells_Samples_PCG_Trigger_Alt_Median=Cells_Samples_PCG_Trigger_Alt_Median[Condition_Global]
Cells_Samples_NICHybNCG_Trigger_Alt_Median=Cells_Samples_NICHybNCG_Trigger_Alt_Median[Condition_Global]
Cells_Samples_PICHybNCG_Trigger_Alt_Median=Cells_Samples_PICHybNCG_Trigger_Alt_Median[Condition_Global]
Cells_Samples_Dual_ICHybNCG_Trigger_Alt_Median=Cells_Samples_Dual_ICHybNCG_Trigger_Alt_Median[Condition_Global]
Cells_Samples_PIC_Trigger_Alt_Median=Cells_Samples_PIC_Trigger_Alt_Median[Condition_Global]
Cells_Samples_NIC_Trigger_Alt_Median=Cells_Samples_NIC_Trigger_Alt_Median[Condition_Global]
Cells_Samples_Dual_IC_Trigger_Alt_Median=Cells_Samples_Dual_IC_Trigger_Alt_Median[Condition_Global]
Cells_Samples_NOMET_Trigger_Alt_Median=Cells_Samples_NOMET_Trigger_Alt_Median[Condition_Global]
Cells_Samples_NOMET_V_Extent_Median=Cells_Samples_NOMET_V_Extent_Median[Condition_Global]
Cells_Samples_NCG_Multistrokes=Cells_Samples_NCG_Multistrokes[Condition_Global]
Cells_Samples_ID=Cells_Samples_ID[Condition_Global]
Cells_Samples_Nb_Long_Flashes=Cells_Samples_Nb_Long_Flashes[Condition_Global]
Cells_Samples_Lon=Cells_Samples_Lon[Condition_Global]
Cells_Samples_Lat=Cells_Samples_Lat[Condition_Global]
Cells_Samples_Hour=Cells_Samples_Hour[Condition_Global]

Cells_Samples_R_Occup_Median=Cells_Samples_R_Occup_Median[Condition_Global]
Cells_Samples_Nb_Potential_DNL=Cells_Samples_Nb_Potential_DNL[Condition_Global]
Cells_Samples_Nb_Potential_DPL=Cells_Samples_Nb_Potential_DPL[Condition_Global]
Cells_Samples_DPL_Alt_Std=Cells_Samples_DPL_Alt_Std[Condition_Global]
Cells_Samples_DNL_Alt_Std=Cells_Samples_DNL_Alt_Std[Condition_Global]
Cells_Samples_Nb_Flashes_ACLR=Cells_Samples_Nb_Flashes_ACLR[Condition_Global]
Cells_Samples_Conf_R=Cells_Samples_Conf_R[Condition_Global]
Cells_Samples_Polarity=Cells_Samples_Polarity[Condition_Global]


############### Filter on all flashes, need to remove flashes inside previoulsy filtered samples 
All_Flashes_Duration=All_Flashes_Duration[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Trigger_Alt=All_Flashes_Trigger_Alt[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Class_Main=All_Flashes_Class_Main[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Class_Complete=All_Flashes_Class_Complete[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Flag_f_ACLR=All_Flashes_Flag_f_ACLR[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_R_Occup= All_Flashes_R_Occup[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Lat=All_Flashes_Lat[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Lon=All_Flashes_Lon[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Nb_S=All_Flashes_Nb_S[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]
All_Flashes_Samples_ID=All_Flashes_Samples_ID[np.isin(All_Flashes_Samples_ID,Cells_Samples_ID)]

NB_SAMPLES=len(Cells_Samples_Time)
print(NB_SAMPLES_before-NB_SAMPLES, np.round((NB_SAMPLES_before-NB_SAMPLES)/NB_SAMPLES_before*100,2),'% of samples filtered')
print('Total #Samples', NB_SAMPLES)
   
print('#flashes after filter:',len(All_Flashes_Class_Main)) 
print('#flashes used by chargepol after filter:',len(All_Flashes_Class_Main[All_Flashes_Flag_f_ACLR==1]))


############################################################################

NB_SAMPLES=len(Cells_Samples_Time)
print(NB_SAMPLES_before-NB_SAMPLES, np.round((NB_SAMPLES_before-NB_SAMPLES)/NB_SAMPLES_before*100,2),'% of samples filtered')
print('Total #Samples', NB_SAMPLES)
   


Cells_Samples_Nb_CG=Cells_Samples_Nb_PCG+Cells_Samples_Nb_NCG 


binx =np.arange(0,15,0.5)
biny=np.arange(0,15,0.5)
ret_count=stats.binned_statistic_2d(Cells_Samples_DPL_Alt,Cells_Samples_DNL_Alt,Cells_Samples_Nb_Flashes, 'count', bins=[binx, biny],expand_binnumbers=True)
Counts_Stats=ret_count.statistic
################################################ HIST 2D Samples distribution ############################## 

Nb_Samples_In_Bins_Mini=10

fig= plt.figure(figsize=(9,7.5))  #(X,Y)  #9 to gave a square with the colorbar at the right
#fig= plt.figure()  #(X,Y)
ax=fig.add_subplot(111)
binx =np.arange(0,15,0.5)
biny=np.arange(0,15,0.5)
ret_count=stats.binned_statistic_2d(Cells_Samples_DPL_Alt,Cells_Samples_DNL_Alt,Cells_Samples_Nb_Flashes, 'count', bins=[binx, biny],expand_binnumbers=True)
Bounds_F= [1,Nb_Samples_In_Bins_Mini,15,20,25,50,75,100,125,137]

#Bounds_F= [Nb_Samples_In_Bins_Mini,2,3,4,5,10,15,20]
norm_F = matplotlib.colors.BoundaryNorm(Bounds_F,cmap_density.N)
Counts_Stats=ret_count.statistic
ax.pcolormesh(binx,biny,np.where((Counts_Stats.T<Nb_Samples_In_Bins_Mini) & (Counts_Stats.T>0),Counts_Stats.T,np.nan),cmap=cmap_density,norm=norm_F,alpha=0.75)
p=ax.pcolormesh(binx,biny,np.where(Counts_Stats.T>=Nb_Samples_In_Bins_Mini,Counts_Stats.T,np.nan),cmap=cmap_density,norm=norm_F,alpha=1)
T_Contour_Flash=stats.binned_statistic_2d(Cells_Samples_DPL_Alt,Cells_Samples_DNL_Alt,Cells_Samples_Nb_Flashes, 'sum', bins=[binx, biny])
Levels_Contour=[10,100,1000,5000]
cp = ax.contour(binx[0:-1]+0.25, biny[0:-1]+0.25,T_Contour_Flash.statistic.T, colors='black', linestyles='dashed',levels=Levels_Contour)
ax.clabel(cp, inline=True,levels=Levels_Contour,fmt="%g")
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.ylabel('Dominant negative layer altitude [$km$]')
plt.xlabel('Dominant positive layer altitude [$km$]')
plt.plot(np.arange(0,16,0.5),np.arange(0,16,0.5),':',color='black')
plt.xlim(0,15)
plt.ylim(0,15)
plt.grid(ls='--',alpha=0.5,which='both')
plt.colorbar(p,label='#Samples per 0.5km bins',ticks=Bounds_F)
plt.tight_layout()
NAME_PLOT='SAMPLES_Distrib_2D_'+domain+'.jpeg'
plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight") 
plt.close()

############################# Filter of samples in 2D bins with less than 10 samples ###
#Ay,Ax=np.where(ret_count.statistic>=Nb_Samples_In_Bins_Mini)
Ay,Ax=np.where(ret_count.statistic>=Nb_Samples_In_Bins_Mini)
Filtered_Bins=np.array([Ax,Ay]).T 
Samples_Binned=ret_count.binnumber.T-1  #artefact i manually correct
Samples_Binned=np.roll(Samples_Binned,1,axis=1)
Samples_Filter= (Samples_Binned[:, None] == Filtered_Bins).all(-1).any(-1) #mask on all pairs of All_bins in b (True if in b) 



########################################## PLOTS of 2D hist median average flash rate per flash type with box plot #################  
def plot_Avg_FR_type(Cells_Samples_Nb_Type,Bounds_FR,Samples_Filter,Type_str,Fmax,Levels_Contour):
    Cells_Samples_FR_Avg=Cells_Samples_Nb_Type/10
    #Bounds_FR=[0,0.1,0.25,0.5,0.75,1,1.5,2,3,4,5,10]
    condition=Samples_Filter & (Cells_Samples_Nb_Type!=0)
    binx =np.arange(0,15,0.5)
    biny=np.arange(0,15,0.5)
    fig= plt.figure(figsize=(12.5,10))  #(X,Y) (15,10)
    #fig= plt.figure()
    
    ax=fig.add_subplot(111)
    #ax.text(0.01,0.98,'a)', fontsize=20,verticalalignment='top', horizontalalignment='left',transform=ax.transAxes)
    ret=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[condition],Cells_Samples_DNL_Alt[condition],Cells_Samples_FR_Avg[condition], 'median', bins=[binx, biny])
    T_Contour_Flash=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[condition],Cells_Samples_DNL_Alt[condition],Cells_Samples_Nb_Type[condition], 'sum', bins=[binx, biny])
    #Levels_Contour=[1,10,100,500,1000]
    cp = ax.contour(binx[0:-1]+0.25, biny[0:-1]+0.25,T_Contour_Flash.statistic.T, colors='black', linestyles='dashed',levels=Levels_Contour)
    ax.clabel(cp, inline=True,levels=Levels_Contour,fmt="%g")
    norm_FR = matplotlib.colors.BoundaryNorm(Bounds_FR,cmap_jet.N)
     #ax.pcolormesh(binx,biny,np.where((ret_count.statistic.T<Nb_Samples_In_Bins_Mini) & (ret_count.statistic.T>0),ret.statistic.T,np.nan),cmap=cmap_jet,norm=norm_F,alpha=0.25)
    p=ax.pcolormesh(binx,biny,np.where(ret_count.statistic.T>=Nb_Samples_In_Bins_Mini,ret.statistic.T,np.nan),cmap=cmap_jet,norm=norm_FR,alpha=1)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #plt.xlabel('DPL altitude [$km$]')
    #plt.ylabel('DNL altitude [$km$]')
     #plt.title('Inter sample median of the 10$^{th}$FR within samples',pad=20)
    #plt.title('Median of samples average '+Type_str+' flash rate', pad=20)
    plt.xlim(1.5,12.5) 
    plt.ylim(1.5,12.5)
    plt.xticks(color='w')
    plt.yticks(color='w')
    plt.plot(binx,biny,'-.',color='black')
    plt.grid(ls='--',alpha=0.5,which='both')
    divider = make_axes_locatable(ax)
    ax_t = divider.append_axes("top", size="5%", pad=0.65)
    plt.colorbar(p,ticks=Bounds_FR,label=' [$f.min^{-1}$]',format="%g", cax=ax_t,orientation='horizontal')
    #plt.colorbar(p,ticks=Bounds_FR,label=' [$f.min^{-1}$]',format="%g",location='top')
    
    
    ax_l=divider.append_axes("left", size="65%", pad=0.24)
    B=[]
    for i in binx:
       B.append(Cells_Samples_FR_Avg[(Cells_Samples_DNL_Alt>=i) & (Cells_Samples_DNL_Alt<i+0.5) & condition & (Cells_Samples_DPL_Alt<Cells_Samples_DNL_Alt)])
    ax_l.boxplot(B,whis=[0,100],whiskerprops={'linestyle': '-.'},vert=False,notch=False,showfliers=False,positions=binx*2+0.5,widths=0.75,patch_artist=True, boxprops=dict(facecolor='white',color='blue'))
    plt.yticks(np.arange(0,len(binx),2),binx[np.arange(0,len(binx),2)].astype(int))
    ax_l.yaxis.set_minor_locator(MultipleLocator(1))
    plt.xscale('log')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel(Type_str+' flash rate \n [$f.min^{-1}$]')
    plt.ylabel('DNL altitude [$km$]')
    #plt.title('Anomalous samples '+Type_str+' flash rate', pad=20)
    plt.xscale('log')
    plt.xlim(0,Fmax)
    plt.ylim(1.5*2,12.5*2)
    plt.grid(ls='--',alpha=0.5)
    
    ax_r=divider.append_axes("right", size="65%", pad=0.24)
    B=[]
    for i in binx:
        B.append(Cells_Samples_FR_Avg[(Cells_Samples_DNL_Alt>=i) & (Cells_Samples_DNL_Alt<i+0.5) & condition & (Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt)])
    ax_r.boxplot(B,whis=[0,100],whiskerprops={'linestyle': '-.'},vert=False,notch=False,showfliers=False,positions=binx*2+0.5,widths=0.75,patch_artist=True, boxprops=dict(facecolor='white',color='red'))
    plt.yticks(np.arange(0,len(binx),2),binx[np.arange(0,len(binx),2)].astype(int))
    ax_r.yaxis.set_minor_locator(MultipleLocator(1))
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel(Type_str+' flash rate \n [$f.min^{-1}$]')
    #plt.title('Normal samples '+Type_str+' flash rate', pad=20)
    #plt.ylabel('DNL altitude [$km$]')
    plt.yticks(color='w')
    plt.xscale('log')
    plt.xlim(0,Fmax)
    plt.ylim(1.5*2,12.5*2)
    plt.grid(ls='--',alpha=0.5)

    ax_b=divider.append_axes("bottom", size="50%", pad=0.18)
    B=[]
    C=[]
    for i in binx:
        B.append(Cells_Samples_FR_Avg[(Cells_Samples_DPL_Alt>=i) & (Cells_Samples_DPL_Alt<i+0.5) & condition & (Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt)])
        C.append(Cells_Samples_FR_Avg[(Cells_Samples_DPL_Alt>=i) & (Cells_Samples_DPL_Alt<i+0.5) & condition & (Cells_Samples_DPL_Alt<Cells_Samples_DNL_Alt)])
    ax_b.boxplot(B,whis=[0,100],whiskerprops={'linestyle': '-.'},positions=binx*2+0.5,showfliers=False,widths=0.75,patch_artist=True, boxprops=dict(facecolor='white',color='red'))
    ax_b.boxplot(C,whis=[0,100],whiskerprops={'linestyle': '-.'},positions=binx*2+0.5,showfliers=False,widths=0.75,patch_artist=True, boxprops=dict(facecolor='white',color='blue'))
    plt.xticks(np.arange(0,len(binx),2),binx[np.arange(0,len(binx),2)].astype(int))
    ax_b.xaxis.set_minor_locator(MultipleLocator(1))
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.ylabel(Type_str+' flash rate \n [$f.min^{-1}$]')
    plt.xlabel('DPL altitude [$km$]')
    plt.yscale('log')
    plt.ylim(0,Fmax-1)
    plt.xlim(1.5*2,12.5*2) 
    plt.grid(ls='--',alpha=0.5)
    # #plt.yticks(color='w')
    # labels=ax_b.get_yticklabels()
    # locs=ax_b.get_yticks()
    # #plt.yticks(color='w')
    # #locs, labels = plt.yticks()
    # #plt.yticks(locs[:-1],labels[:-1],color='k')
    # plt.yticks()
    plt.tight_layout()
    NAME_PLOT='SAMPLES_LAYERS_2D_FR_'+Type_str+'_Avg_'+domain+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight") 
    plt.close()
    return
Bounds_FR=[0,0.1,0.5,1,2,3,5,8,10,15,20,25,30,35,40,50]
Levels_Contour=[10,100,1000,5000]
plot_Avg_FR_type(Cells_Samples_Nb_Flashes,Bounds_FR,Samples_Filter,'Total',100,Levels_Contour)

########################################### PLOTS of +IC fraction, +Cg fraction and IC:CG on Samples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ##############################
def plot_fraction(Cells_Samples_Fraction,Cells_Samples_Nb_Flashes,Fraction_Condition,Bounds_fraction,Fraction_Str,cmap):

    binx =np.arange(0,15,0.5)
    biny=np.arange(0,15,0.5)
    fig= plt.figure(figsize=(9,7.5))  #(X,Y)
    ax=fig.add_subplot(111)
    ret=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[Fraction_Condition],Cells_Samples_DNL_Alt[Fraction_Condition],Cells_Samples_Fraction[Fraction_Condition], 'median', bins=[binx, biny])
    T_Contour_Flash=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[Fraction_Condition],Cells_Samples_DNL_Alt[Fraction_Condition],Cells_Samples_Nb_Flashes[Fraction_Condition], 'sum', bins=[binx, biny])
    norm_Fraction = matplotlib.colors.BoundaryNorm(Bounds_Fraction,cmap.N)
    p=ax.pcolormesh(binx,biny,np.where(ret_count.statistic.T>=Nb_Samples_In_Bins_Mini,ret.statistic.T,np.nan),cmap=cmap,norm=norm_Fraction,alpha=1)
    Levels_Contour=[1,10,100,500,1000]
    cp = ax.contour(binx[0:-1]+0.25, biny[0:-1]+0.25,T_Contour_Flash.statistic.T, colors='black', linestyles='dashed',levels=Levels_Contour)
    ax.clabel(cp, inline=True,levels=Levels_Contour,fmt="%g")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator()) 
    plt.xlim(1.5,12.5) 
    plt.ylim(1.5,12.5)

    plt.ylabel('DNL altitude [$km$]')
    plt.xlabel('DPL altitude [$km$]')
    plt.plot(binx,biny,'-.',color='black')
    plt.grid(ls='--',alpha=0.5,which='both')
    plt.colorbar(p,ticks=Bounds_Fraction,label=Fraction_Str,format="%g")
    # divider = make_axes_locatable(ax)
    # ax_t = divider.append_axes("top", size="5%", pad=0.65)
    # plt.colorbar(p,ticks=Bounds_Fraction,label=Fraction_Str,format="%g", cax=ax_t,orientation='horizontal')
    plt.tight_layout()
    NAME_PLOT='SAMPLES_LAYERS_2D_'+Fraction_Str+'_'+domain+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight") 
    plt.close()
    return



Cells_Samples_ICCG=(Cells_Samples_Nb_IC+Cells_Samples_Nb_NOMET)/Cells_Samples_Nb_CG
ICCG_Condition=(Cells_Samples_Nb_CG!=0) & Samples_Filter
Bounds_Fraction=[0,0.5,1,2,3,5,10,20,30,40,50,60]
plot_fraction(Cells_Samples_ICCG,Cells_Samples_Nb_CG,ICCG_Condition,Bounds_Fraction,'IC-CG Ratio',cmap_density)

Bounds_Fraction=[0,0.25,0.5,1,2,3,5,10,15,20,30,40,50]
Cells_Samples_PCG_Fraction=Cells_Samples_Nb_PCG/Cells_Samples_Nb_CG*100   #samples with at least one cg and part of the binned samples with 10 samples minimum 
PCG_Fraction_condition=(Cells_Samples_Nb_CG!=0) & Samples_Filter
plot_fraction(Cells_Samples_PCG_Fraction,Cells_Samples_Nb_CG,PCG_Fraction_condition,Bounds_Fraction,'Positive CG fraction [%]',cmap_density)

Cells_Samples_PIC_Fraction=Cells_Samples_Nb_PIC/(Cells_Samples_Nb_PIC+Cells_Samples_Nb_NIC)*100
PIC_Fraction_Condition=(Cells_Samples_Nb_PIC+Cells_Samples_Nb_NIC!=0) & Samples_Filter
Bounds_Fraction=[0,1,3,5,10,15,20,40,60,80,90,95,99,100]
plot_fraction(Cells_Samples_PIC_Fraction,Cells_Samples_Nb_PIC+Cells_Samples_Nb_NIC,PIC_Fraction_Condition,Bounds_Fraction,' Positive IC fraction [%]',cmap_density)



#####################################################################################################
######################################## FLASH INITIATION HEIGHT ################################### 
##################################################################################################### 

Theorical_Trigger_Alt=np.copy(Cells_Samples_DPL_Alt)
Theorical_Trigger_Alt[(Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt) & (Samples_Filter)]=(Cells_Samples_DPL_Alt[(Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt) & (Samples_Filter)]-Cells_Samples_DNL_Alt[(Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt) & (Samples_Filter)])/2+Cells_Samples_DNL_Alt[(Cells_Samples_DPL_Alt>Cells_Samples_DNL_Alt) & (Samples_Filter)]
Theorical_Trigger_Alt[(Cells_Samples_DPL_Alt<Cells_Samples_DNL_Alt) & (Samples_Filter)]=(Cells_Samples_DNL_Alt[(Cells_Samples_DPL_Alt<Cells_Samples_DNL_Alt) & (Samples_Filter)]-Cells_Samples_DPL_Alt[(Cells_Samples_DPL_Alt<Cells_Samples_DNL_Alt) & (Samples_Filter)])/2+Cells_Samples_DPL_Alt[(Cells_Samples_DPL_Alt<Cells_Samples_DNL_Alt) & (Samples_Filter)]

################################## Initiation altitude difference with theorical one for each flash type  #######################
def Plot_Flash_Init_Height_Diff(Cells_Samples_Nb_Flashes,Samples_Filter,Cells_Samples_Flash_Trigger_Alt_Median,Flash_Str):
    fig= plt.figure(figsize=(9,7.5))  #(X,Y)
    ax=fig.add_subplot(111)
    Alt_ticks=np.arange(-4,4.5,0.5)
    norm_Alt= matplotlib.colors.BoundaryNorm(Alt_ticks,cmap_div.N)
    condition=Samples_Filter & (Cells_Samples_Nb_Flashes!=0)
    ret=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[condition],Cells_Samples_DNL_Alt[condition],Cells_Samples_Flash_Trigger_Alt_Median[condition]-Theorical_Trigger_Alt[condition], 'median', bins=[binx, biny])
    #ret_FR=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[condition],Cells_Samples_DNL_Alt[condition],Cells_Samples_FR[condition], 'median', bins=[binx, biny])
    T_Contour_Flash=stats.binned_statistic_2d(Cells_Samples_DPL_Alt[condition],Cells_Samples_DNL_Alt[condition],Cells_Samples_Nb_Flashes[condition], 'sum', bins=[binx, biny])
    Levels_Contour=[1,10,100,500,1000,2500]
    cp = ax.contour(binx[0:-1]+0.25, biny[0:-1]+0.25,T_Contour_Flash.statistic.T, colors='black', linestyles='dashed',levels=Levels_Contour)
    ax.clabel(cp, inline=True,levels=Levels_Contour,fmt="%g")
    p=ax.pcolormesh(binx,biny,ret.statistic.T,cmap=cmap_div,norm=norm_Alt)
    # for y in range(len(binx[:-1])):
    #     for x in range(len(binx[:-1])):
    #         if ret_count.statistic[x,y].T>=Nb_Samples_In_Bins_Mini:
    #           ax.text(binx[x] + 0.25, binx[y] + 0.25, '%.1f' % ret_FR.statistic[x, y].T,horizontalalignment='center',verticalalignment='center',color='grey',fontweight='bold',fontsize=11)
    plt.grid(ls='--',alpha=0.5,which='both')
    plt.xlim(1.5,12.5) 
    plt.ylim(1.5,12.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('DPL altitude [$km$]')
    plt.ylabel('DNL altitude [$km$]')
    plt.plot(binx,biny,'-.',color='black')
    #plt.title('+IC',pad=20)
    plt.colorbar(p,ticks=Alt_ticks,label=Flash_Str+' median difference in initiation altitude \n [$km$]',format="%g")
    plt.tight_layout()
    NAME_PLOT='Cells_LAYERS_Triggers_'+Flash_Str+'_'+domain+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight") 
    plt.show()
    plt.close()
    
Plot_Flash_Init_Height_Diff(Cells_Samples_Nb_PIC, Samples_Filter, Cells_Samples_PIC_Trigger_Alt_Median, '+IC')
Plot_Flash_Init_Height_Diff(Cells_Samples_Nb_NIC, Samples_Filter, Cells_Samples_NIC_Trigger_Alt_Median, '-IC')
Plot_Flash_Init_Height_Diff(Cells_Samples_Nb_NCG, Samples_Filter, Cells_Samples_NCG_Trigger_Alt_Median, '-CG')
Plot_Flash_Init_Height_Diff(Cells_Samples_Nb_PCG, Samples_Filter, Cells_Samples_PCG_Trigger_Alt_Median, '+CG')
Plot_Flash_Init_Height_Diff(Cells_Samples_Nb_NOMET, Samples_Filter, Cells_Samples_NOMET_Trigger_Alt_Median, 'NO MET')





