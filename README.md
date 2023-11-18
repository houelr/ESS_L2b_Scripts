# ESS_L2b_Scripts
The different versions of the libraries used for these processes are detailed in Config.pdf file. Python 3.9.6 was used to run these scripts. 

All processing is performed on L2b data (L2 SAETTA + météorage Data, netcdf files) available on this link, link to DOI. Data must be deposited, by day, in a same folder called L2b_SAETTA. In this folder, the 2 GRID files (GRID_XY2D_CORSICA_1km.npz and GRID_LATLON2D_CORSICA_1km.npz) must be present (they are by default in the reposetory L2b_SAETTA available here). The default path organization should be:

L2b_SAETTA/18MMDD/L2b.nc files

L2b_SAETTA/GRID_LATLON_CORSICA_1km.npz

L2b_SAETTA/GRID_XY2D_CORSICA_1km.npz

5 different scripts allow the user to obtain the samples database presented in the article, as well as plotting all the figures for samples and the cell presented as an example. Only 3 scripts need to be run in sequence. 

The RUN.py script launches the ECTA algorithm (ECTA_RUN.py script) and the L2b data extraction (Cell_Data_Extractor_RUN.py script) for each cell over the entire study period. Processing takes place day by day. The user just needs to change the path of the Work Directory (Wdir) at the very beginning of the script so that it points to his directory with all the study days (L2b_SAETTA folder).  ECTA will create a list of cells (in the form of polygons) as well as a graph summarizing, for each day, all the trajectories of the cells obtained. The Cell_Data_Extractor_RUN.py script extracts for each cells, breaks down the cells into samples, classifies the lightning and uses the Chargepol algorithm (Medina et al. 2021) to determine the charge structure. All the variables for each cell are then saved in a Cell_Domain_CellID.npz file, itself in a directory created for each cell. This step can take several hours depending on the user’s PC configuration.  

Next, the Cell_Data_Plot.py script need to be executed after changing the path of the Wdir at the start of the script. This path must lead to the L2b_SAETTA directory. This script produces different graphs for each cell. Here, it is defaulted to cell #2 of the 180726 situation presented in the paper. 

Finally, the Samples_Stats_Plots.py script must be run after changing the Wdir path at the start of the script.  This path must lead to the L2b_SAETTA directory. This script is used to load all the samples from the study period to produce the statistics and graphs presented in the paper.
