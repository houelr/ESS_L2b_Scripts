
"""
Created on Fri Oct  2 11:44:05 2020

@author: hour
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import time
from scipy import stats
import matplotlib
from matplotlib import colors
from matplotlib.dates import num2date, DateFormatter
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import timedelta
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import warnings
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
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
cmap_polarity=colors.ListedColormap(['blue', 'red'])
cmap_jet=plt.get_cmap('jet')
cmap_bwr=plt.get_cmap('bwr')
cmap_PRGN=plt.get_cmap('PRGn')

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
    'font.size': 12, # was 10
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
                        #File reading and parameters
###############################################################

##### ATTENTION SUPPRIME TT LES WARNINGS EN OUTPUT#######"""
H_max=15
warnings.filterwarnings("ignore")

LMA='SAETTA'

Wdir=Path('Path/To/L2b_SAETTA')
DAY='180726'
domain='JULY18'
ONE_CELL=2 ## for all cells --> -1, here will do graphics for cell #2

if LMA=='SAETTA':
    #Wdir=Path('E:\Ronan_work\hour\WORK\L2b\L2b_SAETTA')
    GRID_LATLON2D_CORSICA_1km=np.load(Path(Wdir/'GRID_LATLON2D_CORSICA_1km.npz'))
    LON2D=GRID_LATLON2D_CORSICA_1km['LON2D']  #to get that, i did meshgrid with XD and YD --> 2D array in meter and then m(X2D,Y2D)
    LAT2D=GRID_LATLON2D_CORSICA_1km['LAT2D']
    lat_min_REGION=41 
    lat_max_REGION=43.5
    lon_min_REGION=6.5
    lon_max_REGION=10.6

# #if we want to focus on only one cell 
if ONE_CELL!=-1:
    print('Focus on one cell:')
    CELLS_ID=np.zeros(1).astype(int)
    CELLS_ID[0]=int(ONE_CELL)
else:
    print('PLOT OF PROPERTIES FOR EACH CELL')
    CELLS_ID_file=np.load(Path(Wdir,DAY,'ECTA',domain+'_Cells_ID_list.npz'))
    CELLS_ID=CELLS_ID_file['CELLS_ID_list']
    print('No. CELLS:'+str(len(CELLS_ID)))
    #CELLS_ID=CELLS_ID[20::]
 
for Cell_id in CELLS_ID:            #WARINING masked array can be not well loaded, need to mask again 
#for Cell_id in CELLS_ID[15:]: 
    print('\n'+DAY+ ' - CELL: '+str(Cell_id)+' ('+str(np.round(np.where(CELLS_ID==Cell_id)[0][0]/len(CELLS_ID)*100,1))+'%)')
    Cell_str=Path(DAY,'ECTA',domain+'_Cell_'+str(Cell_id))
    Save_path=Path(Wdir/Cell_str)
    name_cellfile='CELL_'+domain+'_'+str(Cell_id)+'.npz'
    CELLfile = np.load(Path(Save_path,name_cellfile))
    Cell_id=int(CELLfile['Cell_id'])
    Cell_LMA_alt=CELLfile['Cell_LMA_alt']
    Cell_LMA_flash=CELLfile['Cell_LMA_flash']
    Cell_LMA_flash_len=CELLfile['Cell_LMA_flash_len']
    Cell_LMA_lat=CELLfile['Cell_LMA_lat']
    Cell_LMA_lon=CELLfile['Cell_LMA_lon']
    Cell_LMA_power=CELLfile['Cell_LMA_power']
    Cell_LMA_time=CELLfile['Cell_LMA_time']
    Cell_LMA_time_period=CELLfile['Cell_LMA_time_period']
    Cell_MET_current=CELLfile['Cell_MET_current']
    Cell_MET_lat=CELLfile['Cell_MET_lat']
    Cell_MET_lon=CELLfile['Cell_MET_lon']
    Cell_MET_strk_to_LMA=CELLfile['Cell_MET_strk_to_LMA']
    Cell_MET_time=CELLfile['Cell_MET_time']
    Cell_MET_type=CELLfile['Cell_MET_type']
    Cell_MET_quality=CELLfile['Cell_MET_quality']
    Cell_Flash_class_main=CELLfile['Cell_Flash_class_main']
    Cell_Flash_class_complete=CELLfile['Cell_Flash_class_complete']
    Flash_type_id=CELLfile['Flash_type_id']
    Cell_trajectory_LON=CELLfile['Cell_trajectory_LON']
    Cell_trajectory_LAT=CELLfile['Cell_trajectory_LAT']
    Nb_S=CELLfile['Nb_S']   #Number of sources for each flash
    Duration=CELLfile['Duration']      #Duration of each flash 
    Flash_time_period=CELLfile['Flash_time_period'] #time of occurence of each flash
    Flash_first_source_flag=CELLfile['Flash_first_source_flag'] #1 for 1st sources for each flash 0 for others
    Horizontal_Flash_area=CELLfile['Horizontal_Flash_area'] #Area of the convexhull of the horizontal plan projection of VHF sources
    L_H=CELLfile['L_H'] #sqrt of the Area, horizontal'
    Multiplicity=CELLfile['Multiplicity']   # Number of strokes associated to each CG, 0 for no CGs 
    Max_CurrentCG=CELLfile['Max_CurrentCG'] # Current min or max associated to each CG stokes 
    Sum_CurrentCG=CELLfile['Sum_CurrentCG']  #Sum of all positive or negative strokes of all CG, if only one= min or max 
    Current_1st_stroke=CELLfile['Current_1st_stroke'] #Current associated to the first stroke for CG, 0 fo the rest 
    Flash_time_norm=CELLfile['Flash_time_norm']  #normalized according to first and last VHF sources of the cell, 0 for first flash and 1 for last 
    Flash_deltaT=CELLfile['Flash_deltaT'] #Time difference between max stroke associated to a CG and the start of teh flash  
    Delta_T_stroke=CELLfile['Delta_T_stroke'] #Time diff between flash start and first stroke 
    Delta_T_stroke_normed=CELLfile['Delta_T_stroke_normed']  #normed time diff between flash start and first stroke 
    Delta_T_stroke_pulse_normed=CELLfile['Delta_T_stroke_pulse_normed'] #Time diff between first stroke and first pulse 
    Delta_T_stroke_pulse=CELLfile['Delta_T_stroke_pulse'] #Time diff between first stroke and first pulse 
    Flash_vertical_extension=CELLfile['Flash_vertical_extension'] #95th -5th alt for each flash
    Flash_trigger_alt=CELLfile['Flash_trigger_alt']  #mean altitude of the sources in the first 500 mircroseconds of the flash 
    Flash_trigger_time=CELLfile['Flash_trigger_time']  #mean time of the sources in the first 500 mircroseconds of the flash
    Cell_LMA_flash_class_main=CELLfile['Cell_LMA_flash_class_main']
    Cell_LMA_flash_class_complete=CELLfile['Cell_LMA_flash_class_complete']
    F=CELLfile['F']
    
    #ACLR variables  CHARGEPOL
    Flag_f_ACLR=CELLfile['Flag_f_ACLR']
    ACLR_Polarity_layer=CELLfile['ACLR_Polarity_layer']
    ALT_AC=CELLfile['ALT_AC']
    Tresh_list=CELLfile['Tresh_list']
    L_speed_list=CELLfile['L_speed_list']
    Tresh_alt_10_up_list=CELLfile['Tresh_alt_10_up_list']
    Tresh_alt_10_down_list=CELLfile['Tresh_alt_10_down_list']
    Tresh_alt_90_down_list=CELLfile['Tresh_alt_90_down_list']
    Tresh_alt_90_up_list=CELLfile['Tresh_alt_90_up_list']
    F_type=CELLfile['F_type']
    MSE_list=CELLfile['MSE_list']
    R2_list=CELLfile['R2_list']
    F_ACLR_1stprocess=CELLfile['F_ACLR']
    Len_time_window=CELLfile['Len_time_window']
    c_PB=CELLfile['c_PB']
    Layer_height_pos=CELLfile['Layer_height_pos']
    Layer_height_neg=CELLfile['Layer_height_neg']
    Cell_PL=CELLfile['Cell_PL']
    Cell_NL=CELLfile['Cell_NL']
    Cell_PL_Sample=CELLfile['Cell_PL_Sample']
    Cell_NL_Sample=CELLfile['Cell_NL_Sample']
    Cell_time_interval=CELLfile['Cell_time_interval']
    Cell_DPL_Alt_interval=CELLfile['Cell_DPL_Alt_interval']
    Cell_DNL_Alt_interval=CELLfile['Cell_DNL_Alt_interval']
    Samples_Polarity=CELLfile['Samples_Polarity']
    Cell_PL_interval_max=CELLfile['Cell_PL_interval_max']
    Cell_NL_interval_max=CELLfile['Cell_NL_interval_max']
    Z=CELLfile['Z']

    FED_TIME=CELLfile['FED_TIME'] 
    FED_5min=CELLfile['FED_5min']
    FED_5min=np.ma.masked_where(FED_5min<=0,FED_5min)
    FED_EXTRACT_TIME=CELLfile['FED_EXTRACT_TIME']
    ZD=CELLfile['ZD']
    FED_LON_ALT=CELLfile['FED_LON_ALT']
    FED_ALT_LAT=CELLfile['FED_ALT_LAT']
    FED_LON_LAT=CELLfile['FED_LON_LAT']
    FED_TIME_ALT=CELLfile['FED_TIME_ALT']
    print('LMA cell start: '+Convert_time_UTC(Cell_LMA_time[0])+'-'+'End:'+Convert_time_UTC(Cell_LMA_time[-1]))
    Cell_duration=int(Cell_LMA_time_period[-1]-Cell_LMA_time_period[0]) #in minutes 
    print("LMA cell duration: "+str(Cell_duration)+" min")
    
        
    lat_min=np.min(Cell_LMA_lat)-0.1
    lat_max=np.max(Cell_LMA_lat)+0.1
    lon_min=np.min(Cell_LMA_lon)-0.1
    lon_max=np.max(Cell_LMA_lon)+0.1
    
    nb_flash=len(np.unique(Cell_LMA_flash))
    nb_CG=len(Cell_Flash_class_main[(Cell_Flash_class_main=='+CG') |(Cell_Flash_class_main=='-CG')])
    m_cell = Basemap(projection='merc',resolution='h',fix_aspect=True,suppress_ticks=False,llcrnrlat=lat_min,urcrnrlat=lat_max,llcrnrlon=lon_min,urcrnrlon=lon_max)
    Cell_LMA_X,Cell_LMA_Y=m_cell(Cell_LMA_lon,Cell_LMA_lat) #in meter
    Cell_MET_X,Cell_MET_Y=m_cell(Cell_MET_lon,Cell_MET_lat)#in meter
    X_min,Y_min=m_cell(lon_min,lat_min)
    X_max,Y_max=m_cell(lon_max,lat_max)


 
    
    print("Before Chargepol PLOTS --- %s seconds ---" % (time.time() - start_time))
    ############################################### Chargepol PLOTS #################################################"
    Tau=(10*10**-3)/(3600*24)
    
    
    Cell_NO_ACLR_L=np.zeros(len(np.arange(0,H_max,0.5)))
    for f in np.unique(F[Flag_f_ACLR==0]):
        L=np.histogram(Cell_LMA_alt[(ACLR_Polarity_layer==0) & (Cell_LMA_flash==f)]/1000.,np.arange(0,H_max+0.5,0.5))
        L=L[0]
        L=np.where(L>0,1,0)
        Cell_NO_ACLR_L=Cell_NO_ACLR_L+L
  
    for t in range(0,len(Cell_time_interval),1):
        fig= plt.figure(figsize=(7.5,7.5))  #(X,Y) 
        ax=fig.add_subplot(111)
        plt.barh(Z,width=Cell_PL_Sample[t,:],height=0.5,color='red',alpha=0.5,align='edge')
        plt.barh(Z,width=Cell_NL_Sample[t,:],height=0.5,color='blue',alpha=0.5,align='edge')
        plt.title('Altitude of sample dominant charge layers - Cell'+str(Cell_id)+'-'+DAY+'\n '+Convert_time_UTC(Cell_time_interval[t])+' UTC Sample',pad=20)
        plt.xlabel('# Flashes per 0.5km bin')
        plt.ylabel('Altitude [$km$]')
        plt.ylim(0,15)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.grid()
        plt.tight_layout()
        NAME_PLOT='ACLR_Charge_Layers_Samples_'+Convert_time_UTC(Cell_time_interval[t])+'_'+domain+'_'+str(Cell_id)+'.jpeg'
        plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight")
        plt.close()
          

    #### Distribution of flash per alt-bin and Dominant layer
    Cell_DPL=np.sum(Cell_PL_interval_max,0) #Dominant positive layer 
    Cell_DNL=np.sum(Cell_NL_interval_max,0)
    fig= plt.figure(figsize=(8,7))  #(X,Y)
    ax=fig.add_subplot(131)
    #ax=fig.add_subplot(121)
    plt.barh(Z,width=Cell_PL,height=0.5,color='red',alpha=0.5,align='edge')
    plt.barh(Z,width=Cell_NL,height=0.5,color='blue',alpha=0.5,align='edge')
    plt.barh(Z,width=Cell_NO_ACLR_L,height=0.5,color='grey',alpha=0.2,align='edge')
    #plt.title('Altitude of inferred charge layers - Cell'+str(Cell_id)+'-'+DAY,pad=20)
    plt.xlabel('# Flashes per 0.5km bin')
    plt.ylabel('Altitude [$km$]')
    plt.ylim(0,15)
    plt.xscale('log')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid()
    plt.tight_layout()
    NAME_PLOT='CHARGEPOL_Charge_Layers_Global_Max'+domain+'_'+str(Cell_id)+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight")
    plt.close() 

    Cell_DPL_Alt=np.zeros(len(Cell_time_interval))
    Cell_DNL_Alt=np.zeros(len(Cell_time_interval))
    #Polarity of 10 min samples and type associated 
    fig= plt.figure(figsize=(9,10))  #(X,Y) 
    ax=fig.add_subplot(311)
    for t in range(0,len(Cell_time_interval),1):
        if len(Cell_PL_interval_max[t,:][Cell_PL_interval_max[t,:]!=0])==0:
            ax.axvline(Cell_time_interval[t],ls='--',color='black')
            continue
        Cell_DPL_Alt[t]=np.mean(Z[Cell_PL_interval_max[t,:]==np.max(Cell_PL_interval_max[t,:])])
        Cell_DNL_Alt[t]=np.mean(Z[Cell_NL_interval_max[t,:]==np.max(Cell_NL_interval_max[t,:])])
        
        
        PL_min_alt=np.min(Cell_DPL_Alt_interval[t,:][Cell_DPL_Alt_interval[t,:]!=0])
        PL_max_alt=np.max(Cell_DPL_Alt_interval[t,:][Cell_DPL_Alt_interval[t,:]!=0])+0.5
        NL_min_alt=np.min(Cell_DNL_Alt_interval[t,:][Cell_DNL_Alt_interval[t,:]!=0])
        NL_max_alt=np.max(Cell_DNL_Alt_interval[t,:][Cell_DNL_Alt_interval[t,:]!=0])+0.5
        if t==0:
            coeffxmin=(np.min(Cell_LMA_time)-np.min(Cell_LMA_time))/(np.max(Cell_LMA_time)-np.min(Cell_LMA_time)) #0
        else:
            coeffxmin=(Cell_time_interval[t-1]-np.min(Cell_LMA_time))/(np.max(Cell_LMA_time)-np.min(Cell_LMA_time))
        coeffxmax=(Cell_time_interval[t]-np.min(Cell_LMA_time))/(np.max(Cell_LMA_time)-np.min(Cell_LMA_time))
        
        ax.axvline(Cell_time_interval[t-1],ls='--',color='black',zorder=0)
        ax.axvline(Cell_time_interval[t],ls='--',color='black',zorder=0)
        ax.axhspan(Cell_DNL_Alt[t],Cell_DNL_Alt[t]+0.5,xmin=coeffxmin,xmax=coeffxmax,alpha=0.75,color='blue',zorder=3)
        ax.axhspan(Cell_DPL_Alt[t],Cell_DPL_Alt[t]+0.5,xmin=coeffxmin,xmax=coeffxmax,alpha=0.75,color='red',zorder=3)
    ax.set_ylim(0,H_max)
    ax.set_xlim(np.min(Cell_LMA_time),np.max(Cell_LMA_time))
    plt.title('Inferred dominant dipole charge structure for samples \n Cell'+str(Cell_id)+'-'+DAY,pad=20)
    plt.xlabel('Time UTC')
    plt.ylabel('Altitude [$km$]')
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,10))) #every 20 min
    ax.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,2)))  #every
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #plt.legend()
    plt.tight_layout()
    NAME_PLOT='ACLR_Layers_samples_10min_'+domain+'_'+str(Cell_id)+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight")
    plt.close() 
    
    print("After Chargepol plots --- %s seconds ---" % (time.time() - start_time))
    
    
    ###########################################  FLASH RATE COMPUTATION #########################################"
    
    Cell_duration=int(np.max(Cell_LMA_time_period))-int(np.min(Cell_LMA_time_period))+1
    if Cell_duration%2==1:
        Cell_duration=Cell_duration+1      #not odd for 2min mean
   # Cell_duration=int((np.max(Cell_LMA_time)-np.min(Cell_LMA_time))*24*60+1)
    FR_T=np.zeros(Cell_duration)  #FR total, all flash 
    FR_IC=np.zeros(Cell_duration) #IC
    FR_PCG=np.zeros(Cell_duration) #CG +
    FR_NCG=np.zeros(Cell_duration) #CG -
    FR_Amb=np.zeros(Cell_duration) #Ambiguous
    FR_NO_MET=np.zeros(Cell_duration) #No MET 
    Cell_minute=np.empty(0)
    Cell_TIME=np.empty(0)
    Flash_id_control=np.empty(0)
    Cell_time_minute=np.zeros(Cell_duration)
    Cell_time_minute[0]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_LMA_time[0]).replace(second=0,microsecond=0)+timedelta(minutes=1))
    for i in np.arange(1,Cell_duration,1):
        Cell_time_minute[i]=matplotlib.dates.date2num(matplotlib.dates.num2date(Cell_time_minute[i-1])+timedelta(minutes=1))
    
    T=0
    for t in Cell_time_minute:#for plot, one minute before and after the cell, time asscoiated to the first souce for each flash 
        Time_condition_first_source=(Cell_LMA_time<=t) & (Cell_LMA_time>t-1/(24*60)) & (Flash_first_source_flag==1)
        FR_T[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source]).astype(int)) 
        FR_IC[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source & (Cell_LMA_flash_class_main=='IC')]))
        FR_NCG[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source & (Cell_LMA_flash_class_main=='-CG')]))
        FR_PCG[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source & (Cell_LMA_flash_class_main=='+CG')]))
        FR_Amb[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source & (Cell_LMA_flash_class_main=='Ambiguous')]))
        FR_NO_MET[T]=len(np.unique(Cell_LMA_flash[Time_condition_first_source & (Cell_LMA_flash_class_main=='NO MET')]))
        Flash_id_control=np.append(Flash_id_control,np.unique(Cell_LMA_flash[Time_condition_first_source]).astype(int))
        T=T+1 
    

    
    
    FR_T_Accu=np.zeros(Cell_duration)
    FR_T_Accu[0]=FR_T[0]
    FR_NCG_Accu=np.zeros(Cell_duration)
    FR_NCG_Accu[0]=FR_NCG[0]
    FR_PCG_Accu=np.zeros(Cell_duration)
    FR_PCG_Accu[0]=FR_PCG[0]
    FR_IC_Accu=np.zeros(Cell_duration)
    FR_IC_Accu[0]=FR_IC[0]
    FR_Amb_Accu=np.zeros(Cell_duration)
    FR_Amb_Accu[0]=FR_Amb[0]
    FR_NO_MET_Accu=np.zeros(Cell_duration)
    FR_NO_MET_Accu[0]=FR_NO_MET[0]
    R_NCG_Accu=np.zeros(Cell_duration)
    R_NCG_Accu[0]=FR_NCG_Accu[0]/FR_T_Accu[0]
    R_PCG_Accu=np.zeros(Cell_duration)
    R_PCG_Accu[0]=FR_PCG_Accu[0]/FR_T_Accu[0]
    R_IC_Accu=np.zeros(Cell_duration)
    R_IC_Accu[0]=FR_IC_Accu[0]/FR_T_Accu[0]
    R_Amb_Accu=np.zeros(Cell_duration)
    R_Amb_Accu[0]=FR_Amb_Accu[0]/FR_T_Accu[0]
    R_NO_MET_Accu=np.zeros(Cell_duration)
    R_NO_MET_Accu[0]=FR_NO_MET_Accu[0]/FR_T_Accu[0]
    
    for m in np.arange(1,Cell_duration,1):
        FR_T_Accu[m]=FR_T_Accu[m-1]+FR_T[m]
        FR_NCG_Accu[m]=FR_NCG_Accu[m-1]+FR_NCG[m]
        FR_PCG_Accu[m]=FR_PCG_Accu[m-1]+FR_PCG[m]
        FR_IC_Accu[m]=FR_IC_Accu[m-1]+FR_IC[m]
        FR_Amb_Accu[m]=FR_Amb_Accu[m-1]+FR_Amb[m]
        FR_NO_MET_Accu[m]=FR_NO_MET_Accu[m-1]+FR_NO_MET[m]
        
        #R_CG_Accu[m]=CG_Accu[m]/CGIC_Accu[m]*100.
        R_NCG_Accu[m]=FR_NCG_Accu[m]/FR_T_Accu[m]*100.
        R_PCG_Accu[m]=FR_PCG_Accu[m]/FR_T_Accu[m]*100.
        R_IC_Accu[m]=FR_IC_Accu[m]/FR_T_Accu[m]*100.
        R_Amb_Accu[m]=FR_Amb_Accu[m]/FR_T_Accu[m]*100.
        R_NO_MET_Accu[m]=FR_NO_MET_Accu[m]/FR_T_Accu[m]*100.
    




    print("Start Cell evolution treatment and graph --- %s seconds ---" % (time.time() - start_time))
    ###################################### SUMMARY PLOT OF LIGHTNINGS PARAMETERS EVOLUTION WITH TIME  #############################
    ############## Sources density evolution ###############################
    #fig= plt.figure(figsize=(19.2*3/4,5/2*10.8))  #(X,Y)  
    fig= plt.figure(figsize=(8,15))  #(X,Y) (11,20) 
    ax_density=fig.add_subplot(511)
    #Cell_time_density,t_bins,alt_bins=np.histogram2d(Cell_LMA_time,Cell_LMA_alt/1000,[np.arange(np.min(Cell_LMA_time),np.max(Cell_LMA_time)+1/(24*60),0.5/(24*60)),np.arange(np.min(Cell_LMA_alt/1000),np.max(Cell_LMA_alt/1000)+0.2,0.1)])
    Cell_time_density,t_bins,alt_bins=np.histogram2d(Cell_LMA_time,Cell_LMA_alt/1000,[np.arange(np.min(Cell_LMA_time),np.max(Cell_LMA_time),0.5/(24*60)),np.arange(np.min(Cell_LMA_alt/1000),np.max(Cell_LMA_alt/1000)+0.2,0.1)])
    Cell_time_density= np.ma.masked_where(Cell_time_density==0,Cell_time_density) #POUR AVOIR EN BLANC LES cases avec 0 source
    #ax_density.pcolormesh(t_bins[:-1],alt_bins[:-1],Cell_time_density.T,shading='gouraud',cmap=cmap_density)
    p=ax_density.pcolormesh(t_bins[:-1],alt_bins[:-1],Cell_time_density.T,cmap=cmap_density,norm=matplotlib.colors.LogNorm())
    # m2.pcolormesh(x_bins,y_bins,np.log10(LMA_density_map.T),cmap=cmap_density,zorder=2) #log10 pour une meilleure représentation de la dynamique
    cbar=plt.colorbar(p,label='VHF sources density \n [$.30s^{-1}.100m^{-1}$]',) 
    plt.ylabel('Altitude \n [$km$]')
    ax_density.set_ylim(0,H_max)
    plt.xlim(np.min(Cell_LMA_time)-1/(24*60),np.max(Cell_LMA_time))
    ax_density.yaxis.set_minor_locator(AutoMinorLocator())
    ax_density.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,10))) #every 20 min
    ax_density.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax_density.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,2)))  #every 5 min
    ax_density.text(0.99,0.01,'Log$_{10}$ scale - Max: '+ str(int(np.max(Cell_time_density)))+' $sources.30s^{-1}.100m^{-1}$',transform=ax_density.transAxes,verticalalignment='bottom', horizontalalignment='right')
    #plt.title('Sources density, inferred charge layers, altitude of triggers, flash rate and flash duration with time \n Cell '+str(Cell_id)+' - '+DAY,pad=20) 
    ax_density.text(0.99,0.99,'#Flashes: '+str(len(np.unique(Cell_LMA_flash))),transform=ax_density.transAxes,verticalalignment='top', horizontalalignment='right')
    ##### ACLR classification #####
    ax_polarity=fig.add_subplot(512)
    ax_polarity.scatter(Cell_LMA_time[ACLR_Polarity_layer==0],ALT_AC[ACLR_Polarity_layer==0],color='lightgrey',marker="o",s=s_VHF*2)
    ax_polarity.scatter(Cell_LMA_time[ACLR_Polarity_layer!=0],ALT_AC[ACLR_Polarity_layer!=0],c=ACLR_Polarity_layer[ACLR_Polarity_layer!=0],marker="o",s=s_VHF*coeff,cmap=cmap_polarity)
    ax_polarity.set_ylim(0,H_max)
    plt.ylabel('Altitude \n [$km$]')
    ax_polarity.set_xlim(np.min(Cell_LMA_time)-1/(24*60),np.max(Cell_LMA_time))
    for t in range(0,len(Cell_time_interval),1):
        ax_polarity.axvline(Cell_time_interval[t],ls='--',color='black')
    ax_polarity.yaxis.set_minor_locator(AutoMinorLocator())
    ax_polarity.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,10))) #every 20 min
    ax_polarity.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax_polarity.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,2)))  #every 5 min
    ax_polarity.text(0.01,0.99,str(len(F[Flag_f_ACLR==1]))+' flashes used',transform=ax_polarity.transAxes,verticalalignment='top', horizontalalignment='left')    
    ax_polarity.text(0.99,0.99,str(len(F_type[F_type==1]))+' +ICs  '+str(len(F_type[F_type==-1]))+' -ICs ',transform=ax_polarity.transAxes,verticalalignment='top', horizontalalignment='right')
    #ax_polarity.set_title(str(len(F[Flag_f_ACLR==1]))+' flashes used  '+str(len(F_type[F_type==1]))+' +ICs  '+str(len(F_type[F_type==-1]))+' -ICs ',pad=20)
    #### ALTITUDE OF FLASH TRIGGERS #### 
    ax_trig = fig.add_subplot(513)
    ax_trig.scatter(Flash_trigger_time[Cell_Flash_class_main=='Ambiguous'],Flash_trigger_alt[Cell_Flash_class_main=='Ambiguous'],marker="o",color='gold',label='Ambiguous')
    ax_trig.scatter(Flash_trigger_time[Cell_Flash_class_main=='NO MET'],Flash_trigger_alt[Cell_Flash_class_main=='NO MET'],marker="o",color='grey',label='NO MET')
    ax_trig.scatter(Flash_trigger_time[Cell_Flash_class_main=='IC'],Flash_trigger_alt[Cell_Flash_class_main=='IC'],marker="o",color='black',label='IC')
    ax_trig.scatter(Flash_trigger_time[Cell_Flash_class_main=='-CG'],Flash_trigger_alt[Cell_Flash_class_main=='-CG'],marker="s",color='blue',label='-CG')
    ax_trig.scatter(Flash_trigger_time[Cell_Flash_class_main=='+CG'],Flash_trigger_alt[Cell_Flash_class_main=='+CG'],marker="s",color='red',label='+CG')
    ax_trig.set_ylim(0,H_max)
    plt.ylabel('Altitude \n [$km$]')
    ax_polarity.set_xlabel('Time (UTC)')
    ax_trig.text(0.99,0.99,str(len(F[Cell_Flash_class_complete=='+IC']))+' +ICs  '+str(len(F[Cell_Flash_class_complete=='-IC']))+' -ICs ',transform=ax_trig.transAxes,verticalalignment='top', horizontalalignment='right')
    plt.xlim(np.min(Cell_LMA_time)-1/(24*60),np.max(Cell_LMA_time))
    ax_trig.yaxis.set_minor_locator(AutoMinorLocator())
    ax_trig.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,10))) #every 20 min
    ax_trig.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax_trig.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,2)))  #every 5 min
    #ax_trig.legend(loc='lower center', bbox_to_anchor= (0.5,1.01), ncol=5,nlign=2,borderaxespad=0, frameon=False)
    ax_trig.legend()
    ################ FLASH RATE ##########################
    ax1 = fig.add_subplot(514)
    width = -1/(24*60) # the width of the bars
    #ax1.bar(Cell_minute,FR_T,width,label='Flashes', color='navy')
    ax1.bar(Cell_time_minute,FR_PCG,width, color='red',align='edge',edgecolor='black',label='+CG')
    ax1.bar(Cell_time_minute,FR_NCG,width,bottom=FR_PCG, color='blue',align='edge',edgecolor='black',label='-CG')
    ax1.bar(Cell_time_minute,FR_IC,width,bottom=FR_PCG+FR_NCG,color='black',align='edge',edgecolor='black',label='IC')
    ax1.bar(Cell_time_minute,FR_NO_MET,width,bottom=FR_PCG+FR_NCG+FR_IC,color='grey',align='edge',edgecolor='black',label='NO MET')
    ax1.bar(Cell_time_minute,FR_Amb,width,bottom=FR_PCG+FR_NCG+FR_IC+FR_NO_MET,color='gold',align='edge',edgecolor='black',label='Ambiguous')
    plt.ylim(0,np.max(FR_T)+1)
    plt.ylabel('Flash rate \n [$f.min^{-1}$]')
    ax1.yaxis.get_major_locator().set_params(integer=True)
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2 = ax1.twinx() 
    ax2.plot(Cell_time_minute,R_PCG_Accu,color='red',label='+CG R: '+str(np.round(R_PCG_Accu[-1],2))+'$\%$')
    ax2.plot(Cell_time_minute,R_NCG_Accu,color='blue',label='-CG R: '+str(np.round(R_NCG_Accu[-1],2))+'$\%$')
    ax2.plot(Cell_time_minute,R_IC_Accu,color='black',label='IC R: '+str(np.round(R_IC_Accu[-1],2))+'$\%$')
    ax2.plot(Cell_time_minute,R_NO_MET_Accu,color='grey',label='NO MET R: '+str(np.round(R_NO_MET_Accu[-1],2))+'$\%$')
    ax2.plot(Cell_time_minute,R_Amb_Accu,color='gold',label='Amb. R: '+str(np.round(R_Amb_Accu[-1],2))+'$\%$') 
    plt.xlim(np.min(Cell_LMA_time)-1/(24*60),np.max(Cell_LMA_time))   
    #ax2.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax2.set_ylim(0,65)
    ax2.set_ylabel('Ratio [$\%$]')
    #ax2.legend(loc='lower center', bbox_to_anchor= (0.5, 1.01), ncol=5,borderaxespad=0, frameon=False)
    ax2.legend()
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,10))) #every 20 min
    ax2.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax2.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,2)))  #every 5 min
    ax1.set_xlabel('Time (UTC)')
    
    ########## Flashes duration evolution ##############
    ax_fdur = fig.add_subplot(515)
    ax_fdur.scatter(Flash_trigger_time[Cell_Flash_class_main=='Ambiguous'],Duration[Cell_Flash_class_main=='Ambiguous'],marker="o",color='gold',label='Ambiguous')
    ax_fdur.scatter(Flash_trigger_time[Cell_Flash_class_main=='NO MET'],Duration[Cell_Flash_class_main=='NO MET'],marker="o",color='grey',label='NO MET')
    ax_fdur.scatter(Flash_trigger_time[Cell_Flash_class_main=='IC'],Duration[Cell_Flash_class_main=='IC'],marker="o",color='black',label='IC')
    ax_fdur.scatter(Flash_trigger_time[Cell_Flash_class_main=='-CG'],Duration[Cell_Flash_class_main=='-CG'],marker="s",color='blue',label='-CG')
    ax_fdur.scatter(Flash_trigger_time[Cell_Flash_class_main=='+CG'],Duration[Cell_Flash_class_main=='+CG'],marker="s",color='red',label='+CG')
    #ax_fdur.set_ylim()
    plt.ylabel('Duration \n [$s$]')
    plt.xlim(np.min(Cell_LMA_time)-1/(24*60),np.max(Cell_LMA_time)) #one minute before first flash and one minute after last one   
    ax_fdur.yaxis.set_minor_locator(AutoMinorLocator())
    plt.yscale('log')
    ax_fdur.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,10))) #every 20 min
    ax_fdur.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax_fdur.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,2)))  #every 5 min
    plt.xlabel('Time UTC')
    plt.tight_layout()
    NAME_PLOT='Cell_Evolution_'+str(Cell_id)+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight")
    plt.close() 
    
    

    
    print("Before 3D VHF sources --- %s seconds ---" % (time.time() - start_time)) 
    ############### 3D TIME CELL LMA + MET ######################
    #fig= plt.figure(figsize=(19.2,10.8))  #(X,Y)  
    s_VHF=1
    fig= plt.figure(figsize=(10,10))  #(X,Y)    
 
     # #######cartopy version 
    ax_main = fig.add_subplot(projection=ccrs.Mercator())
    ax_main.coastlines(resolution='10m')
    ax_main.scatter(Cell_LMA_lon,Cell_LMA_lat,marker="o",s=s_VHF,c=Cell_LMA_time,cmap=cmap_jet,label='VHF sources',transform=ccrs.PlateCarree())
    ax_main.set_extent([lon_min,lon_max,lat_min,lat_max])
    gl=ax_main.gridlines(draw_labels=True,linestyle='--',color='gray',alpha=0.5)
    gl.xlines = True
    gl.ylines = True
    gl.xlabels_top=False
    gl.ylabels_right=False
    gl.xlocator = tck.MaxNLocator(nbins=5,min_n_ticks=3,steps=None)
    gl.ylocator = tck.MaxNLocator(nbins=5,min_n_ticks=3,steps=None)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    divider = make_axes_locatable(ax_main) # for right and top plot, same size as main map  
    
    ax_r_alt =divider.append_axes("right", size="37%", pad="10%",axes_class=plt.Axes)  #right plot
    #ax_r_alt =divider.append_axes("right", size="37%", pad="8%",axes_class=plt.Axes)  #right plot
    ax_r_alt.scatter(Cell_LMA_alt/1000,Cell_LMA_lat,marker="o",s=s_VHF,c=Cell_LMA_time,cmap=cmap_jet) 
    ax_r_alt.set_xlim(0,H_max)
    ax_r_alt.tick_params(labelbottom=True, labelleft=False)
    ax_r_alt.yaxis.set_minor_locator(AutoMinorLocator())
    ax_r_alt.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Altitude \n [$km$]')
    ax_r_alt.set_ylim(lat_min,lat_max)
    
    ax_u_alt =divider.append_axes("top", size="25%", pad="7%",axes_class=plt.Axes)  # top plot
    ax_u_alt.scatter(Cell_LMA_lon,Cell_LMA_alt/1000,marker="o",s=s_VHF,c=Cell_LMA_time,cmap=cmap_jet) 
    ax_u_alt.set_xlim(lon_min,lon_max)
    ax_u_alt.set_ylim(0,H_max)
    plt.ylabel('Altitude \n [$km$]')
    ax_u_alt.yaxis.set_minor_locator(AutoMinorLocator())
    ax_u_alt.xaxis.set_minor_locator(AutoMinorLocator())
    ax_u_alt.tick_params(labelbottom=False, labelleft=True)
    
    plt.draw()
    P_u_alt=ax_u_alt.get_position().bounds
    P_r_alt=ax_r_alt.get_position().bounds
    
    ax_hist=fig.add_axes([P_r_alt[0], P_u_alt[1], P_r_alt[2], P_u_alt[3]]) #rect  (x0, y0, width, height)  #hist plot 
    #plt.xlabel('Alt Distribution')
    ax_hist.set_ylim(0,H_max)
    density = stats.kde.gaussian_kde(Cell_LMA_alt/1000)
    x=np.arange(0,H_max,0.1)
    ax_hist.yaxis.set_minor_locator(AutoMinorLocator())
    ax_hist.xaxis.set_minor_locator(AutoMinorLocator())
    # ax_hist.hist(Cell_LMA_alt/1000,bins=H_max*10,orientation='horizontal', density=True, color='navy',histtype='step')
    ax_hist.plot(density(x),x,color='navy')
    #ax_hist.set_yticks([])
    ax_hist.grid()
            
    ax_time=fig.add_axes([P_u_alt[0], P_u_alt[1]+P_u_alt[3]+0.07, (P_r_alt[0]+P_r_alt[2])-P_u_alt[0], P_u_alt[3]]) #rect  (x0, y0, width, height
    ax_time.scatter(Cell_LMA_time,Cell_LMA_alt/1000,marker="o",s=s_VHF,c=Cell_LMA_time,vmin=np.min(Cell_LMA_time),vmax=np.max(Cell_LMA_time),cmap=cmap_jet) 
    ax_time.set_ylim(0,H_max)
    ax_time.set_xlim(np.min(Cell_LMA_time)-30/(24*3600),np.max(Cell_LMA_time)+30/(24*3600))
    plt.ylabel('Altitude \n [$km$]')
    #ax_time.set_xlabel('Time (UTC)')
    #plt.xlabel('Time UTC [$hh:mm$]')
    ax_time.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,20))) #every 20 min
    ax_time.xaxis.set_major_formatter(DateFormatter("%H%M")) # need to be before autolocator
    ax_time.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,60,5)))  #every 5 min
    
    ax_time.yaxis.set_minor_locator(AutoMinorLocator())
    #plt.title('SAETTA and MET data - Cell '+str(Cell_id)+' - '+DAY,pad=20)  
    NAME_PLOT='CELL_LMA_MET_TIME_'+domain+'_'+str(Cell_id)+'.jpeg'
    plt.savefig(Path(Save_path/NAME_PLOT),dpi=300,bbox_inches="tight")
    plt.close()     
            
print(" END --- %s seconds ---" % (time.time() - start_time)) 