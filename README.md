# ESS L2b scripts 

### Overview

This repository contains scripts for the analysis of LMA SAETTA lightning data and Météorage data using the different python scrits. The different versions of the libraries used for these processes are detailed in the **Config.pdf** file. The scripts are written in Python 3.9.6.

### Data Source

All processing is performed on L2b data (L2 SAETTA + météorage Data, netcdf files) available at the following link: https://doi.org/10.25326/542#v02. The data must be deposited, by day, in a folder named `L2b_SAETTA` along with the two GRID files, namely `GRID_XY2D_CORSICA_1km.npz` and `GRID_LATLON2D_CORSICA_1km.npz` of this git repository. The default path organization should be as follows:

```
L2b_SAETTA/18MMDD/L2b.nc
L2b_SAETTA/GRID_LATLON_CORSICA_1km.npz
L2b_SAETTA/GRID_XY2D_CORSICA_1km.npz
```

### Scripts

There are five different scripts provided to obtain the samples database presented in the article and to generate various plots. Only three scripts need to be run in sequence.

1. **RUN.py:** This script launches the ECTA algorithm (`ECTA_RUN.py` script) and the L2b data extraction (`Cell_Data_Extractor_RUN.py` script) for each cell over the entire study period. Processing takes place day by day. *The user needs to change the path of the Work Directory (Wdir)* at the beginning of the script to point to their directory with all the study days (`L2b_SAETTA` folder). ECTA will create a list of cells (in the form of polygons) as well as a graph summarizing, for each day, all the trajectories of the cells obtained. The `Cell_Data_Extractor_RUN.py` script extracts, breaks down the cells into samples, classifies the lightning, and uses the Chargepol algorithm (Medina et al. 2021) to determine the charge structure. All the variables for each cell are then saved in a `Cell_Domain_CellID.npz` file, itself in a directory created for each cell. This step can take several hours depending on the user’s PC configuration.

2. **Cell_Data_Plot.py:** After changing the path of the **Wdir** at the start of the script, this script produces different graphs for each cell. The default is set to cell #2 of the 180726 situation presented in the paper.

3. **Samples_Stats_Plots.py:** This script must be run after changing the **Wdir** path at the start of the script. The path should lead to the `L2b_SAETTA` directory. This script is used to load all the samples from the study period to produce the statistics and graphs presented in the paper.
