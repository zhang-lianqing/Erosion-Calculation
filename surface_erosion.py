# ==========================================================================================================
# Name:      surface_erosion.py
# Purpose:     
#            1. Calibrate surface feature-based erosion models:
#               - E = f(slope(S)), from Montgmoery and Brandon (2002) and Roering et al. (1999)
#               - E = f(slope(S), precipitation(P))
#               - E = f(slope(S), precipitation(P), vegetation cover(V))
#            2. Use the calibrated erosin model to calculate the erosion rates for the catchment.
#
# Author: Lianqing Zhang (lianqing.zhang@uni-jena.de)
# 
#           This algorithm combines the cosmogenic nuclide-based and surface feature-based erosion models
#               to constrain the surface erosion rate distribution for a catchment.
# 
# Created:      2023.12
# Copyright:   
# ==========================================================================================================

import arcpy
import os
import sys
import numpy as np
import pandas as pd
import tempfile
import shutil
from scipy.optimize import curve_fit

# Get the current directory (cur_dir) of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))
#  then, add it into the system path
sys.path.append(cur_dir)
#  to access the data base and unitilty functions.
from data_base import *
from utility_functions import *

# ==========================================================================================================

# Check licenses.
arcpy.CheckOutExtension('Spatial')
# clean up in memeory
arcpy.Delete_management("in_memory")

# ==========================================================================================================

# setting environment
arcpy.env.overwriteOutput = True
# setting temporary folder
temp_folder = tempfile.mkdtemp()

# ==========================================================================================================

# Get Parameters.

Input_catchments_shapefile = arcpy.GetParameterAsText(0)
Input_DEM_raster = arcpy.GetParameterAsText(1)
Density = float(arcpy.GetParameterAsText(2))
# CNs = arcpy.GetParameterAsText(3) # find it in data_base.py
Concentration_field = arcpy.GetParameterAsText(4)
Concentration_uncertainty_field = arcpy.GetParameterAsText(5)
Shielding_factor_field = arcpy.GetParameterAsText(6)
Shielding_factor_raster = arcpy.GetParameterAsText(7)
Model = arcpy.GetParameterAsText(8)
Background_erosion_field = arcpy.GetParameterAsText(9) # unit: mm/kyr
Critical_slope = float(arcpy.GetParameterAsText(10))
Precipitation_intensity_field = arcpy.GetParameterAsText(11)
Precipitation_intensity_raster = arcpy.GetParameterAsText(12)
# a = float(arcpy.GetParameterAsText(13)) # find it in data_base.py
Vegetation_cover_field = arcpy.GetParameterAsText(14)
Vegetation_cover_raster = arcpy.GetParameterAsText(15)
Output_folder_for_erosion_raster = arcpy.GetParameterAsText(16)
Output_folder_for_k_N_pairs = arcpy.GetParameterAsText(17)

# ==========================================================================================================

# Model == 
# "E = f(slope)"
# "E = f(slope, precipitation)"
# "E = f(slope, precipitation, vegetation cover)"

# add centroid coordinate for each catchment.
arcpy.management.CalculateGeometryAttributes(Input_catchments_shapefile, [['cent_long', 'CENTROID_X'], ['cent_lat', 'CENTROID_Y']])

# add fields
if Model == "E = f(slope)":
    k_field = 'k_S'
    E_field = "E_S"
    unc_kplus_field = "unc_kp_S"
    unc_kminus_field = "unc_km_S"
    unc_Eplus_field = "unc_Ep_S"
    unc_Eminus_field = "unc_Em_S"
elif Model == "E = f(slope, precipitation)":
    k_field = 'k_SP'
    E_field = 'E_SP'
    unc_kplus_field = "unc_kp_SP"
    unc_kminus_field = "unc_km_SP"
    unc_Eplus_field = "unc_Ep_SP"
    unc_Eminus_field = "unc_Em_SP"
elif Model == "E = f(slope, precipitation, vegetation cover)":
    k_field = 'k_SPV'
    E_field = 'E_SPV'
    unc_kplus_field = "unc_kp_SPV"
    unc_kminus_field = "unc_km_SPV"
    unc_Eplus_field = "unc_Ep_SPV"
    unc_Eminus_field = "unc_Em_SPV"
else:
    arcpy.AddError('The input model name is not correct!')

arcpy.AddField_management(Input_catchments_shapefile, k_field, "FLOAT", "", "", 50)
arcpy.AddField_management(Input_catchments_shapefile, E_field, 'FLOAT', '', '', 50)
arcpy.AddField_management(Input_catchments_shapefile, unc_kplus_field, 'FLOAT', '', '', 50)
arcpy.AddField_management(Input_catchments_shapefile, unc_kminus_field, 'FLOAT', '', '', 50)
arcpy.AddField_management(Input_catchments_shapefile, unc_Eplus_field, 'FLOAT', '', '', 50)
arcpy.AddField_management(Input_catchments_shapefile, unc_Eminus_field, 'FLOAT', '', '', 50)

# ==========================================================================================================

if Model == "E = f(slope)":

    field_name = ["SHAPE@", "FID", "cent_long", "cent_lat", Concentration_field, Concentration_uncertainty_field,
                  Background_erosion_field, Shielding_factor_field, k_field, E_field, unc_kplus_field, 
                  unc_kminus_field, unc_Eplus_field, unc_Eminus_field]

    with arcpy.da.UpdateCursor(Input_catchments_shapefile, field_name) as cursor:
            # process each catchment
            for row in cursor:

                # get attributes
                geometry = row[0]
                FID = row[1]
                centroid_long = row[2]
                centroid_lat = row[3]
                concentration_CN = row[4]
                concentration_uncertainty = row[5]
                background_erosion_field = row[6] / 10000 # convert unit from mm/kyr to cm/yr
                shielding_factor_field = row[7]

                # get the utm projected coordinate system
                utm_spatial_reference = get_utm_project_name(centroid_long, centroid_lat)

                # get the catchment's DEM
                DEM = arcpy.sa.ExtractByMask(Input_DEM_raster, geometry, 'INSIDE')
                # project DEM
                DEM_project = os.path.join(temp_folder, f"DEM_project_{FID}.tif")
                arcpy.management.ProjectRaster(DEM, DEM_project, utm_spatial_reference)
                # calculate slope
                SLP = arcpy.sa.SurfaceParameters(DEM_project,"SLOPE")

                # get DEM array.
                DEM_array = arcpy.RasterToNumPyArray(DEM_project, nodata_to_value=-9999)
                #  convert the DEM array into float data type.
                DEM_array = DEM_array.astype(float)
                #  then, conver the -9999 to NoData.
                DEM_array[DEM_array == -9999] = np.nan
                # get slope array
                SLP_array = arcpy.RasterToNumPyArray(SLP, nodata_to_value=np.nan)
                # get latitude array
                lat_array = get_coordinates_array(DEM, DEM_array)

                # get shielding factor array
                if Shielding_factor_raster:
                    shielding_raster = arcpy.sa.ExtractByMask(Shielding_factor_raster, geometry, 'INSIDE')
                    # project
                    shielding_raster_project = os.path.join(temp_folder, f"shielding_raster_project_{FID}.tif")
                    arcpy.management.ProjectRaster(shielding_raster, shielding_raster_project, utm_spatial_reference)
                    # get shielding factor array
                    shieldingfactor_array = arcpy.RasterToNumPyArray(shielding_raster_project, nodata_to_value=-9999)
                    shieldingfactor_array = shieldingfactor_array.astype(float)
                    shieldingfactor_array[shieldingfactor_array == -9999] = np.nan
                    # check shape
                    if not np.array_equal(shieldingfactor_array.shape, DEM_array.shape):
                        arcpy.AddError('Shielding factor raster has different shape from DEM.')
                    shieldingfactor = shieldingfactor_array
                else:
                    shieldingfactor = shielding_factor_field
                
                # Calculate initial k.
                # prepare elevation values
                elevation_small = np.nanpercentile(DEM_array, 15)
                elevation_large = np.nanpercentile(DEM_array, 75)
                # prepare slope values
                slope_small = np.nanpercentile(SLP_array, 25)
                slope_large = np.nanpercentile(SLP_array, 85)

                # to avoid negative initial k, we set:
                if slope_large >= Critical_slope:
                    slope_large = Critical_slope * 0.85

                # calculate the small k
                k_small = calculate_k_for_Montgomery_and_Brandon(centroid_lat, centroid_long, elevation_small,
                                                            concentration_CN, Density, slope_large, Critical_slope, shielding_factor_field)
                # calculate the large k
                k_large = calculate_k_for_Montgomery_and_Brandon(centroid_lat, centroid_long, elevation_large,
                                                            concentration_CN, Density, slope_small, Critical_slope, shielding_factor_field)

                # get initial k values
                initial_k = np.linspace(k_small, k_large, 6)
                
                # calculate physical erosion-weighted catchment average CNs concentration.
                #  create two empty lists for N and k
                N_value = []
                k_value = []
                for k in initial_k:
                    N = calculate_N_use_Montgomery_and_Brandon(centroid_lat, centroid_long, lat_array, k, background_erosion_field,
                                                            SLP_array, DEM_array, Density, Critical_slope, shieldingfactor)
                    N_value.append(N)
                    k_value.append(k)

                # get k, N pair
                df_k_N = pd.DataFrame({
                    'k' : k_value,
                    'N' : N_value })
                df_k_N['k'] = df_k_N['k']*10000 # convert the unit of k from cm/yr to mm/kyr
                # save the k and N pairs (optional).
                if Output_folder_for_k_N_pairs:
                    output_path = Output_folder_for_k_N_pairs + '\\' + f'Model_S_FID_{FID}.csv'
                    df_k_N.to_csv(output_path)
                
                # use power function to fit the k and N.
                #  first normalize the k and N
                k_value = np.array(k_value)
                N_value = np.array(N_value)
                k_normalized = k_value / np.max(k_value)
                N_normalized = N_value / np.max(N_value)
                #  second, use 'curve_fit' to fit
                parameters, covariance = curve_fit(power_function, k_normalized, N_normalized)
                #  get the parameters
                parameter_a, parameter_b = parameters

                # calculate the k for measured nuclide concentration N at catchment outlet.
                #  first normolize the measured CNs concentration
                concentration_CN_normalized = concentration_CN / np.max(N_value)
                #  then, calculate the normalized k and the reverse normalization of k.
                #  *10000 for unit conversion from cm/yr to mm/kyr
                calibrated_k = ((concentration_CN_normalized / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000 

                # add the calibrated k into the catchment attribute table.
                row[8] = calibrated_k

                # calculate the mean erosion rate using the calibrated erosion model.
                erosion_array = background_erosion_field * 10000 + calibrated_k * SLP_array / (1 - (SLP_array / Critical_slope) ** 2)
                row[9] = np.nanmean(erosion_array)

                # output the generated erosion raster (optional).
                if Output_folder_for_erosion_raster: 
                    describe = arcpy.Describe(SLP)
                    lower_left_corner = arcpy.Point(describe.extent.XMin, describe.extent.YMin)
                    # generate erosion raster
                    erosion_raster = arcpy.NumPyArrayToRaster(erosion_array, lower_left_corner, x_cell_size = SLP, y_cell_size=SLP)
                    # define projection
                    arcpy.management.DefineProjection(erosion_raster, utm_spatial_reference)
                    # save to folder
                    output_path = Output_folder_for_erosion_raster + "\\" + f'Model_S_FID_{FID}.tif'
                    erosion_raster.save(output_path)
                
                # calculate uncertainty
                calibrated_k_small = ((((concentration_CN+concentration_uncertainty) / np.max(N_value)) / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000
                calibrated_k_large = ((((concentration_CN-concentration_uncertainty) / np.max(N_value)) / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000
                row[10] = calibrated_k_large - calibrated_k
                row[11] = calibrated_k - calibrated_k_small
                erosion_array_small = background_erosion_field * 10000 + calibrated_k_small * SLP_array / (1 - (SLP_array / Critical_slope) ** 2)
                erosion_array_large = background_erosion_field * 10000 + calibrated_k_large * SLP_array / (1 - (SLP_array / Critical_slope) ** 2)
                row[12] = np.nanmean(erosion_array_large)-np.nanmean(erosion_array)
                row[13] = np.nanmean(erosion_array)-np.nanmean(erosion_array_small)

                cursor.updateRow(row)

# ==========================================================================================================
                    
elif Model == "E = f(slope, precipitation)":

    field_name = ["SHAPE@", "FID", "cent_long", "cent_lat", Concentration_field, Concentration_uncertainty_field, Precipitation_intensity_field,
                   Shielding_factor_field, k_field, E_field, unc_kplus_field, unc_kminus_field, unc_Eplus_field, unc_Eminus_field]

    with arcpy.da.UpdateCursor(Input_catchments_shapefile, field_name) as cursor:
        # process each catchment
        for row in cursor:

            # get attributes
            geometry = row[0]
            FID = row[1]
            centroid_long = row[2]
            centroid_lat = row[3]
            concentration_CN = row[4]
            concentration_uncertainty = row[5]
            precipitation_field = row[6]
            shielding_factor_field = row[7]
            
            # get the projected coordinate system of this catchment
            utm_spatial_reference = get_utm_project_name(centroid_long, centroid_lat)

            # get the catchment's DEM
            DEM = arcpy.sa.ExtractByMask(Input_DEM_raster, geometry, 'INSIDE')
            # project DEM
            DEM_project = os.path.join(temp_folder, f"DEM_project_{FID}.tif")
            arcpy.management.ProjectRaster(DEM, DEM_project, utm_spatial_reference)
            # calculate slope
            SLP = arcpy.sa.SurfaceParameters(DEM_project,"SLOPE")

            # get DEM array.
            DEM_array = arcpy.RasterToNumPyArray(DEM_project, nodata_to_value=-9999)
            DEM_array = DEM_array.astype(float)
            DEM_array[DEM_array == -9999] = np.nan
            # get slope array
            SLP_array = arcpy.RasterToNumPyArray(SLP, nodata_to_value=np.nan)
            # get latitude array
            lat_array = get_coordinates_array(DEM, DEM_array)

            if Shielding_factor_raster:
                shielding_raster = arcpy.sa.ExtractByMask(Shielding_factor_raster, geometry, 'INSIDE')
                # project
                shielding_raster_project = os.path.join(temp_folder, f"shielding_raster_project_{FID}.tif")
                arcpy.management.ProjectRaster(shielding_raster, shielding_raster_project, utm_spatial_reference)
                # get shielding factor array
                shieldingfactor_array = arcpy.RasterToNumPyArray(shielding_raster_project, nodata_to_value=-9999)
                shieldingfactor_array = shieldingfactor_array.astype(float)
                shieldingfactor_array[shieldingfactor_array == -9999] = np.nan
                # checking np array's shape
                if not np.array_equal(shieldingfactor_array.shape, DEM_array.shape):
                    arcpy.AddError('Shielding factor raster has different shape from DEM.')
                shieldingfactor = shieldingfactor_array
            else:
                shieldingfactor = shielding_factor_field
            
            if Precipitation_intensity_raster:
                precipitation_intensity = arcpy.sa.ExtractByMask(Precipitation_intensity_raster, geometry, 'INSIDE')
                # project
                precipitation_intensity_project = os.path.join(temp_folder, f"precipitation_intensity_project_{FID}.tif")
                arcpy.management.ProjectRaster(precipitation_intensity, precipitation_intensity_project, utm_spatial_reference)
                # convert raster to nparray.
                precipitation_intensity_array = arcpy.RasterToNumPyArray(precipitation_intensity_project, nodata_to_value=-9999)
                precipitation_intensity_array = precipitation_intensity_array.astype(float)
                precipitation_intensity_array[precipitation_intensity_array == -9999] = np.nan
                # checking shape
                if not np.array_equal(precipitation_intensity_array.shape, DEM_array.shape):
                    arcpy.AddError('Precipitation intensity raster has different shape from DEM!')
                precipitation = precipitation_intensity_array
            else:
                precipitation = precipitation_field
            
            # Calculate inital k.
            # prepare elevation
            elevation_small = np.nanpercentile(DEM_array, 25)
            elevation_large = np.nanpercentile(DEM_array, 75)
            # prepare slope
            slope_small = np.nanpercentile(SLP_array, 25)
            slope_large = np.nanpercentile(SLP_array, 75)

            # calculate the small k
            k_small = calculate_k_for_Zhang_S_P(centroid_lat, centroid_long, elevation_small,
                                                        concentration_CN, Density, precipitation_field, slope_large, shielding_factor_field)
            # calculate the large k
            k_large = calculate_k_for_Zhang_S_P(centroid_lat, centroid_long, elevation_large,
                                                        concentration_CN, Density, precipitation_field, slope_small, shielding_factor_field)

            # get initial k values
            initial_k = np.linspace(k_small, k_large, 6)

            # calculate N for each initial k
            #  create an empty list for N and k
            N_value = []
            k_value = []
            for k in initial_k:
                # calculate erosion-weighted CN concentration.
                N = calculate_N_use_Zhang_S_P(centroid_lat, centroid_long, lat_array, k, precipitation,
                                                SLP_array, DEM_array, Density, shieldingfactor)
                N_value.append(N)
                k_value.append(k)

            # get k, N pair
            df_k_N = pd.DataFrame({
                'k' : k_value,
                'N' : N_value
            })
            df_k_N['k'] = df_k_N['k']*10000 # convert the unit of k from cm/yr to mm/kyr
            # save the k and N pairs (optional).
            if Output_folder_for_k_N_pairs:
                output_path = Output_folder_for_k_N_pairs + '\\' + f'Model_SP_FID_{FID}.csv'
                df_k_N.to_csv(output_path)

            # use power function to fit the k and N.
            #  first normalize the k and N
            k_value = np.array(k_value)
            N_value = np.array(N_value)
            k_normalized = k_value / np.max(k_value)
            N_normalized = N_value / np.max(N_value)
            #  second, use 'curve_fit' to fit
            parameters, covariance = curve_fit(power_function, k_normalized, N_normalized)
            #  get the parameters
            parameter_a, parameter_b = parameters

            # calculate the corresponding k for measured nuclide concentration N at catchment outlet.
            #  first normolize measured CNs concentration
            concentration_CN_normalized = concentration_CN / np.max(N_value)
            #  then, calculate the normalized k and the reverse normalization of k.
            calibrated_k = ((concentration_CN_normalized / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000

            # add the calibrated k into the catchment attribute table.
            row[8] = calibrated_k 

            # Calculate the mean erosion rate using the calibrated erosion model.
            radian_slope = np.radians(SLP_array)
            sin_slope_array = np.sin(radian_slope)
            erosion_array = calibrated_k * (precipitation ** sin_slope_array)
            # write mean erosion into attribute table.
            row[9] = np.nanmean(erosion_array)

            # output the generated erosion raster (optional).
            if Output_folder_for_erosion_raster: 
                describe = arcpy.Describe(SLP)
                lower_left_corner = arcpy.Point(describe.extent.XMin, describe.extent.YMin)
                erosion_raster = arcpy.NumPyArrayToRaster(erosion_array, lower_left_corner, x_cell_size = SLP, y_cell_size=SLP)
                arcpy.management.DefineProjection(erosion_raster, utm_spatial_reference)
                # output
                output_path = Output_folder_for_erosion_raster + "\\" + f'Model_SP_FID_{FID}.tif'
                erosion_raster.save(output_path)

            # calculate uncertainty
            calibrated_k_small = ((((concentration_CN+concentration_uncertainty) / np.max(N_value)) / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000
            calibrated_k_large = ((((concentration_CN-concentration_uncertainty) / np.max(N_value)) / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000
            row[10] = calibrated_k_large - calibrated_k
            row[11] = calibrated_k - calibrated_k_small
            erosion_array_small = calibrated_k_small * (precipitation ** sin_slope_array)
            erosion_array_large = calibrated_k_large * (precipitation ** sin_slope_array)
            row[12] = np.nanmean(erosion_array_large)-np.nanmean(erosion_array)
            row[13] = np.nanmean(erosion_array)-np.nanmean(erosion_array_small)

            cursor.updateRow(row)

# ==========================================================================================================
                
elif Model == "E = f(slope, precipitation, vegetation cover)":

    field_name = ["SHAPE@", "FID", "cent_long", "cent_lat", Concentration_field, Concentration_uncertainty_field, Precipitation_intensity_field,
                   Vegetation_cover_field, Shielding_factor_field, k_field, E_field, unc_kplus_field, unc_kminus_field, unc_Eplus_field, unc_Eminus_field]
    
    with arcpy.da.UpdateCursor(Input_catchments_shapefile, field_name) as cursor:
        # process each catchment
        for row in cursor:

            # get attributes
            geometry = row[0]
            FID = row[1]
            centroid_long = row[2]
            centroid_lat = row[3]
            concentration_CN = row[4]
            concentration_uncertainty = row[5]
            precipitation_field = row[6]
            vegetation_field = row[7]
            shielding_factor_field = row[8]
            
            # get the projected coordinate system of this catchment
            utm_spatial_reference = get_utm_project_name(centroid_long, centroid_lat)

            # get the catchment's DEM
            DEM = arcpy.sa.ExtractByMask(Input_DEM_raster, geometry, 'INSIDE')
            # project DEM
            DEM_project = os.path.join(temp_folder, f"DEM_project_{FID}.tif")
            arcpy.management.ProjectRaster(DEM, DEM_project, utm_spatial_reference)
            # calculate slope
            SLP = arcpy.sa.SurfaceParameters(DEM_project,"SLOPE")

            # get DEM array.
            DEM_array = arcpy.RasterToNumPyArray(DEM_project, nodata_to_value=-9999)
            DEM_array = DEM_array.astype(float)
            DEM_array[DEM_array == -9999] = np.nan
            # get slope array
            SLP_array = arcpy.RasterToNumPyArray(SLP, nodata_to_value=np.nan)
            # get latitude array
            lat_array = get_coordinates_array(DEM, DEM_array)

            if Shielding_factor_raster:
                shielding_raster = arcpy.sa.ExtractByMask(Shielding_factor_raster, geometry, 'INSIDE')
                # project
                shielding_raster_project = os.path.join(temp_folder, f"shielding_raster_project_{FID}.tif")
                arcpy.management.ProjectRaster(shielding_raster, shielding_raster_project, utm_spatial_reference)
                # get shielding factor array
                shieldingfactor_array = arcpy.RasterToNumPyArray(shielding_raster_project, nodata_to_value=-9999)
                shieldingfactor_array = shieldingfactor_array.astype(float)
                shieldingfactor_array[shieldingfactor_array == -9999] = np.nan
                # checking np array's shape
                if not np.array_equal(shieldingfactor_array.shape, DEM_array.shape):
                    arcpy.AddError('Shielding factor raster has different shape from DEM.')
                shieldingfactor = shieldingfactor_array
            else:
                shieldingfactor = shielding_factor_field
            
            if Precipitation_intensity_raster:
                precipitation_intensity = arcpy.sa.ExtractByMask(Precipitation_intensity_raster, geometry, 'INSIDE')
                # project
                precipitation_intensity_project = os.path.join(temp_folder, f"precipitation_intensity_project_{FID}.tif")
                arcpy.management.ProjectRaster(precipitation_intensity, precipitation_intensity_project, utm_spatial_reference)
                # convert raster to nparray.
                precipitation_intensity_array = arcpy.RasterToNumPyArray(precipitation_intensity_project, nodata_to_value=-9999)
                precipitation_intensity_array = precipitation_intensity_array.astype(float)
                precipitation_intensity_array[precipitation_intensity_array == -9999] = np.nan
                # checking shape
                if not np.array_equal(precipitation_intensity_array.shape, DEM_array.shape):
                    arcpy.AddError('Precipitation intensity raster has different shape from DEM.')
                precipitation = precipitation_intensity_array
            else:
                precipitation = precipitation_field

            if Vegetation_cover_raster:
                vegetation_cover = arcpy.sa.ExtractByMask(Vegetation_cover_raster, geometry, 'INSIDE')
                # project
                vegetation_project = os.path.join(temp_folder, f"vegetation_project_{FID}.tif")
                arcpy.management.ProjectRaster(vegetation_cover, vegetation_project, utm_spatial_reference)
                # convert raster to nparray
                vegetation_array = arcpy.RasterToNumPyArray(vegetation_project, nodata_to_value=-9999)
                vegetation_array = vegetation_array.astype(float)
                vegetation_array[vegetation_array == -9999] = np.nan
                # convert
                vegetation_array = vegetation_array * 2 / 255 - 1
                # set NDVI<0 to NDVI=0
                vegetation_array[vegetation_array < 0] = 0
                # checking shape
                if not np.array_equal(vegetation_array.shape, DEM_array.shape):
                    arcpy.AddError('vegetation cover raster has different shape from DEM!')
                vegetation = vegetation_array
            else:
                vegetation = vegetation_field
            
            # Calculate inital k.
            # prepare elevation
            elevation_small = np.nanpercentile(DEM_array, 25)
            elevation_large = np.nanpercentile(DEM_array, 75)
            # prepare slope
            slope_small = np.nanpercentile(SLP_array, 25)
            slope_large = np.nanpercentile(SLP_array, 75)

            # calculate the small k
            k_small = calculate_k_for_Zhang_S_P_V(centroid_lat, centroid_long, elevation_small,
                                                  concentration_CN, Density, precipitation_field, slope_large, vegetation_field, shielding_factor_field)
            # calculate the large k
            k_large = calculate_k_for_Zhang_S_P_V(centroid_lat, centroid_long, elevation_large,
                                                  concentration_CN, Density, precipitation_field, slope_small, vegetation_field, shielding_factor_field)

            # get initial k values
            initial_k = np.linspace(k_small, k_large, 6)

            # calculate erosion-weighted CN concentration for each initial k.
            #  create an empty list for N and k
            N_value = []
            k_value = []
            for k in initial_k:
                N = calculate_N_use_Zhang_S_P_V(centroid_lat, centroid_long, lat_array, k, precipitation, 
                                                SLP_array, DEM_array, Density, vegetation, shieldingfactor)
                N_value.append(N)
                k_value.append(k)

            # get k, N pair
            df_k_N = pd.DataFrame({
                'k' : k_value,
                'N' : N_value
            })
            df_k_N['k'] = df_k_N['k']*10000 # convert the unit of k from cm/yr to mm/kyr
            # save the k and N pairs (optional).
            if Output_folder_for_k_N_pairs:
                output_path = Output_folder_for_k_N_pairs + '\\' + f'Model_SPV_FID_{FID}.csv'
                df_k_N.to_csv(output_path)

            # use power function to fit the k and N.
            #  first normalize the k and N
            k_value = np.array(k_value)
            N_value = np.array(N_value)
            k_normalized = k_value / np.max(k_value)
            N_normalized = N_value / np.max(N_value)
            #  second, use 'curve_fit' to fit
            parameters, covariance = curve_fit(power_function, k_normalized, N_normalized)
            #  get the parameters
            parameter_a, parameter_b = parameters

            # calculate the corresponding k for measured nuclide concentration N at catchment outlet.
            #  first normolize measured CNs concentration
            concentration_CN_normalized = concentration_CN / np.max(N_value)
            #  then, calculate the normalized k and the reverse normalization of k.
            calibrated_k = ((concentration_CN_normalized / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000

            # add the calibrated k into the catchment attribute table.
            row[9] = calibrated_k

            # Calculate the mean erosion rate using the calibrated erosion model.
            radian_slope = np.radians(SLP_array)
            sin_slope_array = np.sin(radian_slope)
            erosion_array = calibrated_k * (a**vegetation) * (precipitation ** sin_slope_array)

            # write mean erosion into attribute table.
            row[10] = np.nanmean(erosion_array)

            # output the generated erosion raster (optional).
            if Output_folder_for_erosion_raster: 
                describe = arcpy.Describe(SLP)
                lower_left_corner = arcpy.Point(describe.extent.XMin, describe.extent.YMin)
                erosion_raster = arcpy.NumPyArrayToRaster(erosion_array, lower_left_corner, x_cell_size = SLP, y_cell_size=SLP)
                arcpy.management.DefineProjection(erosion_raster, utm_spatial_reference)
                # output
                output_path = Output_folder_for_erosion_raster + "\\" + f'Model_SPV_FID_{FID}.tif'
                erosion_raster.save(output_path)

            # calculate uncertainty
            calibrated_k_small = ((((concentration_CN+concentration_uncertainty) / np.max(N_value)) / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000
            calibrated_k_large = ((((concentration_CN-concentration_uncertainty) / np.max(N_value)) / parameter_a) ** (1 / parameter_b)) * np.max(k_value) * 10000
            row[11] = calibrated_k_large - calibrated_k
            row[12] = calibrated_k - calibrated_k_small
            erosion_array_small = calibrated_k_small * (a**vegetation) * (precipitation ** sin_slope_array)
            erosion_array_large = calibrated_k_large * (a**vegetation) * (precipitation ** sin_slope_array)
            row[13] = np.nanmean(erosion_array_large)-np.nanmean(erosion_array)
            row[14] = np.nanmean(erosion_array)-np.nanmean(erosion_array_small)

            cursor.updateRow(row)

# clean up
shutil.rmtree(temp_folder)
arcpy.Delete_management("in_memory")

