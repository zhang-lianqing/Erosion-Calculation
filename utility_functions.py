# ==================================================================================================
# Name:         utility_functions.py
# Purpose:     
#               Providing functions for surface_erosion.py
#
# Author: Lianqing Zhang (lianqing.zhang@uni-jena.de)
#               Some of the codes are revised based on the BASINGA codes by Charreau et al. (2019).
# 
# Created:      2023.12
# Copyright:   
# ==================================================================================================

import arcpy
import os
import sys
import tempfile
import numpy as np
from scipy import interpolate

# Get the current directory (cur_dir) of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))
#  then, add it into the system path
sys.path.append(cur_dir)
#  to access the data base.
from data_base import *

# Check related licenses.
arcpy.CheckOutExtension('Spatial')
# Setting:
arcpy.env.overwriteOutput = True
# setting temporary folder
temp_folder = tempfile.mkdtemp()

# ==================================================================================================

def get_coordinates_array(DEM, DEM_array):
    """
    Get the latitude array of the catchment. This array is used to calculate the Stone coefficient array.

    Input: 
        DEM: catchment DEM
        DEM_array: it comes from the projected DEM.
    Output:
        lat_array: the latitude array of the catchment.
    """
    desc = arcpy.Describe(DEM) # for getting the extent on latitude dimension
    lat_list = np.linspace(desc.extent.YMax, desc.extent.YMin, DEM_array.shape[0])
    lat_array = np.array(lat_list).reshape(-1, 1)
    lat_array = np.tile(lat_array, DEM_array.shape[1])

    return lat_array


def get_Stone_coefficients(latitude):
    """
    References: (Stone, 2000) and (Lal, 1991).

    Input:
        latitude [degrees], either a np.array or a float.
    Outputs:
        a, b, c, d, e : Stone coefficients.
    Note: these factors are used to calculate the spallation (sp) scalling factors only.
            this function is revised from the BASINGA codes by Charreau et al. (2019).
    """
    latitude = abs(latitude) # use the absolute value of latitude in case the study area is in southern hemisphere.

    if isinstance(latitude, np.ndarray):
        latitude_max = latitude.max()
    else:
        latitude_max = latitude

    if latitude_max < 10:
        dec = 10 - latitude
        a = (dec * St[0, 1] + (10 - dec) * St[1, 1]) / 10
        b = (dec * St[0, 2] + (10 - dec) * St[1, 2]) / 10
        c = (dec * St[0, 3] + (10 - dec) * St[1, 3]) / 10
        d = (dec * St[0, 4] + (10 - dec) * St[1, 4]) / 10
        e = (dec * St[0, 5] + (10 - dec) * St[1, 5]) / 10
    elif latitude_max < 20:
        dec = 20 - latitude
        a = (dec * St[1, 1] + (10 - dec) * St[2, 1]) / 10
        b = (dec * St[1, 2] + (10 - dec) * St[2, 2]) / 10
        c = (dec * St[1, 3] + (10 - dec) * St[2, 3]) / 10
        d = (dec * St[1, 4] + (10 - dec) * St[2, 4]) / 10
        e = (dec * St[1, 5] + (10 - dec) * St[2, 5]) / 10
    elif latitude_max < 30:
        dec = 30 - latitude
        a = (dec * St[2, 1] + (10 - dec) * St[3, 1]) / 10
        b = (dec * St[2, 2] + (10 - dec) * St[3, 2]) / 10
        c = (dec * St[2, 3] + (10 - dec) * St[3, 3]) / 10
        d = (dec * St[2, 4] + (10 - dec) * St[3, 4]) / 10
        e = (dec * St[2, 5] + (10 - dec) * St[3, 5]) / 10
    elif latitude_max < 40:
        dec = 40 - latitude
        a = (dec * St[3, 1] + (10 - dec) * St[4, 1]) / 10
        b = (dec * St[3, 2] + (10 - dec) * St[4, 2]) / 10
        c = (dec * St[3, 3] + (10 - dec) * St[4, 3]) / 10
        d = (dec * St[3, 4] + (10 - dec) * St[4, 4]) / 10
        e = (dec * St[3, 5] + (10 - dec) * St[4, 5]) / 10
    elif latitude_max < 50:
        dec = 50 - latitude
        a = (dec * St[4, 1] + (10 - dec) * St[5, 1]) / 10
        b = (dec * St[4, 2] + (10 - dec) * St[5, 2]) / 10
        c = (dec * St[4, 3] + (10 - dec) * St[5, 3]) / 10
        d = (dec * St[4, 4] + (10 - dec) * St[5, 4]) / 10
        e = (dec * St[4, 5] + (10 - dec) * St[5, 5]) / 10
    elif latitude_max < 60:
        dec = 60 - latitude
        a = (dec * St[5, 1] + (10 - dec) * St[6, 1]) / 10
        b = (dec * St[5, 2] + (10 - dec) * St[6, 2]) / 10
        c = (dec * St[5, 3] + (10 - dec) * St[6, 3]) / 10
        d = (dec * St[5, 4] + (10 - dec) * St[6, 4]) / 10
        e = (dec * St[5, 5] + (10 - dec) * St[6, 5]) / 10
    else:
        a = St[6, 1]
        b = St[6, 2]
        c = St[6, 3]
        d = St[6, 4]
        e = St[6, 5]

    return a, b, c, d, e

def get_atmospheric_coefficients(latitude, longitude):
    """
    Calculate the atmospheric parameters at a site, which are used to calculate the sp scaling factor.
    Inputs:
        latitude [degrees]: a float
        longitude [degrees]: a float
    Outputs:
        sea_level_P [hPa]   : sea level atmospheric pressure.
        sea_level_T [K]     : sea level atmospheric temperature.
        ALR [K/m]           : Lifton Adiabatic Lapse Rate (ALR).
    Note: this function is adapted from the BASINGA codes by Charreau et al. (2019).
    """
    # convert the negative longitude to positive
    longitude = (longitude + 360) % 360
    
    sea_level_P = float(interpolate.interp2d(ERA40_lon, ERA40_lat, ERA40_meanP).__call__(longitude, latitude))

    sea_level_T = float(interpolate.interp2d(ERA40_lon, ERA40_lat, ERA40_meanT).__call__(longitude, latitude))

    ALR = ALR_factors[0] + ALR_factors[1] * latitude + ALR_factors[2] * latitude ** 2 + ALR_factors[3] * latitude ** 3 \
           + ALR_factors[4] * latitude ** 4 + ALR_factors[5] * latitude ** 5 + ALR_factors[6] * latitude ** 6
    
    ALR = -ALR

    return sea_level_P, sea_level_T, ALR


def calculate_production_rate_point(latitude, longitude, elevation, shielding_factor_field):
    """
    First, calculate the scaling factor (sp, fast muon reaction (fm), and slow muon capture (sm)).
    Then, calculate the production rate by each reaction pathway at the site (point).

    Input:
        latitude [degrees]
        longtitude [degrees]
        elevation [m]
    Output:
        production_rate_point_sp [atoms/g/yr]    :   the production rate by sp at the site.
        production_rate_point_fm [atoms/g/yr]    :   the production rate by fm at the site.
        production_rate_point_sm [atoms/g/yr]    :   the production rate by sm at the site.
    Note: this function calculates the production rates at a point,
            which can service determining the initial k.
    """

    # Get coefficients for sp scaling factor calculation.
    sea_level_P, sea_level_T, ALR = get_atmospheric_coefficients(latitude, longitude)
    a, b, c, d, e = get_Stone_coefficients(latitude)

    # Convert elevation [m] to atmospheric pressure [hPa].
    P = sea_level_P * np.exp(-0.03417 / ALR * (np.log(sea_level_T) - np.log(sea_level_T - ALR * elevation)))

    # Calculate scaling factors.
    #  calculate the scaling factor for sp (Stone, 2000).
    scaling_factor_sp = (a + b * np.exp(-P / 150) + c * P + d * P ** 2 + e * P ** 3)
    #  calculate the scaling factors for fm and sm (Braucher et al., 2011).
    scaling_factor_fm = np.exp((1013.25 - P) / atmospheric_attenuation_fm)
    scaling_factor_sm = np.exp((1013.25 - P) / atmospheric_attenuation_sm)

    # Calculate the production rate at the site.
    #  calculate the sp production rate.
    production_rate_point_sp = production_CN_SLHL_sp * scaling_factor_sp * shielding_factor_field
    #  calculate the fm production rate.
    production_rate_point_fm = production_CN_SLHL_fm * scaling_factor_fm * shielding_factor_field
    #  calculate the sm production rate.
    production_rate_point_sm = production_CN_SLHL_sm * scaling_factor_sm * shielding_factor_field

    return production_rate_point_sp, production_rate_point_fm, production_rate_point_sm


def calculate_erosion_rate_point(latitude, longitude, elevation, concentration_CN, density, shielding_factor_field):
    """
    Calculate the erosion rate using the analytical method (Charreau et al., 2019).
    Input: 
        latitude [degrees]
        longitude [degrees]
        elevation [m]
        concentration_CN [atoms/g]
        density [g/cm3]
    Output:
        erosion_rate [cm/yr]     : erosion rate of the site.
        Note: this function is designed to constrain the initial k.
    """

    # Get the production rate by sp, fm, sm pathways at the site.
    production_rate_point_sp, production_rate_point_fm, production_rate_point_sm = \
    calculate_production_rate_point(latitude, longitude, elevation, shielding_factor_field)

    # Calulate the denudation rate without considering the decay constant.
    # It does not matter, because this function is only used to determine the initial k.
    # And we don't need a very accrate initial k.
    erosion_rate = (attenuation_neutron * (production_rate_point_sp / concentration_CN) + \
                       attenuation_fmuon * (production_rate_point_fm / concentration_CN) + \
                       attenuation_smuon * (production_rate_point_sm / concentration_CN)) / density 
    return erosion_rate


def calculate_k_for_Montgomery_and_Brandon(latitude, longitude, elevation, concentration_CN,
                                            density, slope, critical_slope, shielding_factor_field):
    """
    Calculate the coefficient k for the erosion model of Montgomery and Brandon (2002).
    Input:
        latitude [degrees]
        longitude [degrees]
        elevation [m]
        concentration_CN [atoms/g]
        density [g/cm3]
        slope [degrees]
        critical_slope [degrees]
        shielding_factor_field
    Output:
        k
    """

    # Calculate erosion
    erosion = calculate_erosion_rate_point(latitude, longitude, elevation, concentration_CN, density, shielding_factor_field)

    # Calculate k, here we omit 'background_erosion' to avoid negative initial k value.
    # it does not matter because this is only for providing several initial k values, it will
    # not impact the accuracy of the final results. because we include it when we calculate the N and the final erosion raster.
    k = erosion * (1 - (slope / critical_slope) ** 2) / slope

    return k


def calculate_k_for_Zhang_S_P(latitude, longitude, elevation, concentration_CN, density,
                                  precipitation_field, slope, shielding_factor_field):
    """
    Calculate the coefficient k for the erosion model of Zhang et al. (2024).
    E = f(slope(S), precipitation(P))
    Input: 
        latitude [degrees]
        longitude [degrees]
        elevation [m]
        concentration_CN [atoms/g]
        density [g/cm3]
        precipitation_field [mm/yr]
        slope [degrees]
        shielding_factor_field
    Output:
        k
    Note: this model derives from (Zhang et al., under review).
            and this function uses the sine value of the slope.
    """
    # Calculate denudation
    erosion = calculate_erosion_rate_point(latitude, longitude, elevation, concentration_CN, density, shielding_factor_field)

    # Calculate k
    slope_in_radians = np.radians(slope)
    k = erosion / (precipitation_field ** np.sin(slope_in_radians))

    return k


def calculate_k_for_Zhang_S_P_V(latitude, longitude, elevation, concentration_CN, density,
                                  precipitation_field, slope, vegetation_field, shielding_factor_field):
    """
    Calculate the coefficient k for the erosion model of Zhang et al. (2024).
    E = f(slope(S), precipitation(P), vegetation(V))
    Input: 
        latitude [degrees]
        longitude [degrees]
        elevation [m]
        concentration_CN [atoms/g]
        density [g/cm3]
        precipitation_field [mm/yr]
        slope [degrees]
        vegetation cover [NDIV]
        shielding_factor_field
    Output:
        k
    """
    # Calculate denudation
    erosion = calculate_erosion_rate_point(latitude, longitude, elevation, concentration_CN, density, shielding_factor_field)

    # Calculate k
    slope_in_radians = np.radians(slope)
    k = erosion / ((a**vegetation_field) * (precipitation_field ** np.sin(slope_in_radians)))

    return k

def calculate_production_rate_catchment(latitude, longitude, lat_array, DEM_array, shieldingfactor):
    """
    This function calculates the production rate for the catchment by each reaction pathway.
    Input:
        latitude [degrees]
        longitude [degrees]
        lat_array [degrees]: the latitude np.array of the catchment. 
        DEM_array [m]: numpy array
        shieldingfactor: numpy array, shielding factor of each pxiel.
    Output:
        production_rate_catchment_sp [numpy array] [atoms/g/yr]
        production_rate_catchment_fm [numpy array] [atoms/g/yr]
        production_rate_catchment_sm [numpy array] [atoms/g/yr]
    """

    # Get coefficient for sp scaling factor calculation.
    sea_level_P, sea_level_T, ALR = get_atmospheric_coefficients(latitude, longitude)
    a, b, c, d, e = get_Stone_coefficients(lat_array)

    # Convert DEM array (DEM_array) into atmospheric pressure array (P_array).
    P_array = sea_level_P * np.exp(-0.03417 / ALR * (np.log(sea_level_T) - np.log(sea_level_T - ALR * DEM_array)))

    # Calculate scaling factors.
    #  calculate the scaling factor for sp (Stone, 2000).
    scaling_factor_sp = (a + b * np.exp(-P_array / 150) + c * P_array + d * P_array ** 2 + e * P_array ** 3)
    #  calculate the scaling factor for fm and sm (Braucher et al., 2011).
    scaling_factor_fm = np.exp((1013.25 - P_array) / atmospheric_attenuation_fm)
    scaling_factor_sm = np.exp((1013.25 - P_array) / atmospheric_attenuation_sm)

    # Calculate the production rate for the catchment.    
    #  calculate the sp production rate.
    production_rate_catchment_sp = production_CN_SLHL_sp * scaling_factor_sp * shieldingfactor
    #  calculate the fm production rate. 
    production_rate_catchment_fm = production_CN_SLHL_fm * scaling_factor_fm * shieldingfactor
    #  calculate the sm production rate.
    production_rate_catchment_sm = production_CN_SLHL_sm * scaling_factor_sm * shieldingfactor

    return production_rate_catchment_sp, production_rate_catchment_fm, production_rate_catchment_sm


def calculate_N_use_Montgomery_and_Brandon(latitude, longitude, lat_array, k, background_erosion,
                                     SLP_array, DEM_array, density, critical_slope, shieldingfactor):
    """
    This function calculates physical erosion-weighted average concentration for the catchment.
    Input:
        latitude [degrees]
        longitude [degrees]
        k
        background_erosion [cm/yr]
        SLP_array [degrees]
        DEM_array [m]
        density [g/cm3]
        critical_slope [degrees]
        shieldingfactor
    Output:
        N [atoms/g] : physical erosion-weighted average CNs concentration.
    """
    # Get production rate of different reaction pathways.
    production_rate_catchment_sp, production_rate_catchment_fm, production_rate_catchment_sm = \
    calculate_production_rate_catchment(latitude, longitude, lat_array, DEM_array, shieldingfactor)
    
    # Calculate the erosion distribution in the catchment use the erosion model of Montgomery and Brandon (2002).
    # Note: In this erosion_array, there maybe negative values depending on whether the slope is over the critical slope.
    erosion_array = background_erosion + k * SLP_array / (1 - (SLP_array / critical_slope) ** 2)
    physical_erosion_array = k * SLP_array / (1 - (SLP_array / critical_slope) ** 2)

    # Calculate the erosion weighting factor (on the physical erosion could contribute sediments to the outlet).
    # Be careful the NoData in np array.
    weighting_factor = physical_erosion_array / np.nansum(physical_erosion_array)

    # Calculate the denominator of equation (4) in (Charreau et al., 2019).
    denominator_sp = decay_constant_CN + density * erosion_array / attenuation_neutron
    denominator_fm = decay_constant_CN + density * erosion_array / attenuation_fmuon
    denominator_sm = decay_constant_CN + density * erosion_array / attenuation_smuon

    # Calculate the CNs concentration 
    #  produced by different reaction pathways,
    N_sp = production_rate_catchment_sp / denominator_sp
    N_fm = production_rate_catchment_fm / denominator_fm
    N_sm = production_rate_catchment_sm / denominator_sm
    #  and summarize them.
    N_sum = N_sp + N_fm + N_sm
    
    # Calculate the erosion-weighted average concentration.
    N = np.nansum(N_sum * weighting_factor)

    return N

def calculate_N_use_Zhang_S_P(latitude, longitude, lat_array, k, precipitation,
                                  SLP_array, DEM_array, density, shieldingfactor):
    """
    The function calculates physical erosion-weighted average concentration for the catchment.
    E = f(slope, precipitation)
    Input: 
        latitude [degrees]
        longitude [degrees]
        k
        precipitation [mm/yr] : it can be a nparray or a value.
        SLP_array [degrees]
        DEM_array [m]
        density [g/cm3]
        shieldingfactor
    Output:
        N : erosion-weighted average CNs concentration.
    Note: the erosion model is from Zhang et al. (under review).
            and we use the sine value of the slope.
    """
    # Get production rate of different reaction pathways.
    production_rate_catchment_sp, production_rate_catchment_fm, production_rate_catchment_sm = \
    calculate_production_rate_catchment(latitude, longitude, lat_array, DEM_array, shieldingfactor)
    
    # Calculate the erosion.
    radian_value = np.radians(SLP_array)
    sin_value = np.sin(radian_value)
    erosion_array = k * (precipitation ** sin_value)

    # Calculate the erosion weighting factor.
    weighting_factor = erosion_array / np.nansum(erosion_array)

    # Calculate the denominator of equation (4) in (Charreau et al., 2019).
    denominator_sp = decay_constant_CN + density * erosion_array / attenuation_neutron
    denominator_fm = decay_constant_CN + density * erosion_array / attenuation_fmuon
    denominator_sm = decay_constant_CN + density * erosion_array / attenuation_smuon

    # Calculate the CNs concentration 
    #  produced by different reaction pathways,
    N_sp = production_rate_catchment_sp / denominator_sp
    N_fm = production_rate_catchment_fm / denominator_fm
    N_sm = production_rate_catchment_sm / denominator_sm
    #  and summarize them.
    N_sum = N_sp + N_fm + N_sm
    
    # Calculate the erosion weighted average concentration.
    N = np.nansum(N_sum * weighting_factor)

    return N

def calculate_N_use_Zhang_S_P_V(latitude, longitude, lat_array, k, precipitation,
                                  SLP_array, DEM_array, density, vegetation, shieldingfactor):
    """
    The function calculates physical erosion-weighted average concentration for the catchment.
    E = f(slope, precipitation, vegetation cover)
    Input: 
        latitude [degrees]
        longitude [degrees]
        k
        precipitation [mm/yr] : it can be a nparray or a value.
        SLP_array [degrees]
        DEM_array [m]
        density [g/cm3]
        vegetation [NDVI] [0,1]
        shieldingfactor
    Output:
        N : erosion-weighted average CNs concentration.
    Note: the erosion model is from Zhang et al. (under review).
            and we use the sine value of the slope.
    """
    # Get production rate of different reaction pathways.
    production_rate_catchment_sp, production_rate_catchment_fm, production_rate_catchment_sm = \
    calculate_production_rate_catchment(latitude, longitude, lat_array, DEM_array, shieldingfactor)
    
    # Calculate the erosion.
    radian_value = np.radians(SLP_array)
    sin_value = np.sin(radian_value)
    erosion_array = k * (a**vegetation) * (precipitation ** sin_value)

    # Calculate the erosion weighting factor.
    weighting_factor = erosion_array / np.nansum(erosion_array)

    # Calculate the denominator of equation (4) in (Charreau et al., 2019).
    denominator_sp = decay_constant_CN + density * erosion_array / attenuation_neutron
    denominator_fm = decay_constant_CN + density * erosion_array / attenuation_fmuon
    denominator_sm = decay_constant_CN + density * erosion_array / attenuation_smuon

    # Calculate the CNs concentration 
    #  produced by different reaction pathways,
    N_sp = production_rate_catchment_sp / denominator_sp
    N_fm = production_rate_catchment_fm / denominator_fm
    N_sm = production_rate_catchment_sm / denominator_sm
    #  and summarize them.
    N_sum = N_sp + N_fm + N_sm
    
    # Calculate the erosion weighted average concentration.
    N = np.nansum(N_sum * weighting_factor)

    return N

def power_function(x, a, b):

    """
    Define a power function for the relationship between k and N.
    """
    return  a * x **b

def get_utm_project_name(longitude, latitude):
    """
    get the projected coordinate system (UTM) for the catchment
      based on the entroid coordinate of the catchment.
    Inputs:
        longtitude [degrees]: centroid coordinate
        latitude [degrees]: controid coordinate
    Output:
        the cooresponding projected spatical reference
    """
    # Determine UTM zone based on the input point
    utm_zone_number = int((longitude + 180) / 6) + 1  # Calculate UTM zone based on longitude
    
    # Determine hemisphere (N for Northern Hemisphere, S for Southern Hemisphere)
    hemisphere = 'N' if latitude >= 0 else 'S'
    
    # Create a spatial reference for the UTM zone
    utm_spatial_reference = arcpy.SpatialReference(f"WGS 1984 UTM Zone {utm_zone_number}{hemisphere}")
    
    return utm_spatial_reference

