#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:58:52 2021

@author: hour
"""
import glob
import os, os.path
import numpy as np
from pathlib import Path


###################################################################################
Wdir=Path('Path/To/L2b_SAETTA') #Change here the path to your work directory called L2b_SAETTA with all DAYS
################################## DOMAIN NAME and DATES ###########################

domains=['JUNE18','JULY18','AUGUST18','SEPTEMBER18','OCTOBER18']
DATES=['1806*','1807*','1808*','1809*','1810*'] #all days in june, july etc...
LMA='SAETTA'


for t in range(len(domains)):
    domain=domains[t]
    DAYS_paths=sorted(glob.glob(str(Path(Wdir,DATES[t]))))   
    DAYS=[]
    for i in DAYS_paths:
        DAYS=np.append(DAYS,os.path.basename(i))
    
    ##### ECTA
    import ECTA_RUN
    for DAY in DAYS:
        T_start=0*60
        T_end=23*60
        Continuous_activity=ECTA_RUN.ECTA(DAY,domain,T_start,T_end,LMA,Wdir)
        if Continuous_activity==1:
            T_start=3*60 #Limit for continuous tracking between 2 days
        else:
            T_start=0

for t in range(len(domains)):
    domain=domains[t]
    DAYS_paths=sorted(glob.glob(str(Path(Wdir,DATES[t]))))
    DAYS=[]
    for i in DAYS_paths:
        DAYS=np.append(DAYS,os.path.basename(i))
        
    # #Extractor
    import Cell_Data_Extractor_RUN
    for DAY in DAYS:
        print(DAY)
        Cell_Data_Extractor_RUN.Cell_Data_Extractor(DAY,domain,LMA,Wdir)

