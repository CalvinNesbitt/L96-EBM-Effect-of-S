"""
Contains:
- Locations where we save attractor data.
- Functions for helping us to open attractor data.
"""
########################################################################
## Imports
########################################################################
# Standard package imports
import xarray as xr
import numpy as np
import sys

# Function to identify where we're running
from pathlib import Path
mac_home_dir = Path('/Users/cfn18')
home_dir = Path.home()
def on_mac():
    return mac_home_dir == home_dir

########################################################################
## Data + Custom import Locations
########################################################################

# Data parent adn attractor object directories
if on_mac():
    # PD for Attractor Data: Should be SB, W and M.
    attractor_data_pd = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Effect-of-S/Attractor-Data/'
    attractorObjectDirectory = '/Users/cfn18/Documents/PhD-Work/Third-Year/Instanton-Work/L96-EBM-Instanton-Cleaned-Up/L96-EBM-Effect-of-S/L96-EBM-Path-Object-Classes/'
else:
    print('Please specify location of attractor data.')

sys.path.append(attractorObjectDirectory)
from attractors import L96EBMAttractor
from plots import *

########################################################################
## Attractor File Names
########################################################################

def attractor_folder_name(end_state, pd=attractor_data_pd, S=10):
    "end_state, string: One of w, sb or m."
    return pd + f'/S_{S:.3f}/{end_state.upper()}-State/'.replace('.', '_')

def w_attractor_file_name(pd=attractor_data_pd, S=10):
    sd = attractor_folder_name('w', pd=pd, S=S)
    return sd + '1.nc'

def sb_attractor_file_name(pd=attractor_data_pd, S=10):
    sd = attractor_folder_name('sb', pd=pd, S=S)
    return sd + '1.nc'

def m_state_file_name(pd=attractor_data_pd, S=10):
    sd = attractor_folder_name('m', pd=pd, S=S)
    return sd + '1.nc'

########################################################################
## Functions for opening determinsitic attractors & M-State
########################################################################

def get_attractor(file_name):
    return L96EBMAttractor(file_name)

def get_w_attractor(pd=attractor_data_pd, S=10):
    file_name = w_attractor_file_name(pd=pd, S=S)
    return get_attractor(file_name)

def get_sb_attractor(pd=attractor_data_pd, S=10):
    file_name = sb_attractor_file_name(pd=pd, S=S)
    return get_attractor(file_name)

def get_m_state(pd=attractor_data_pd, S=10):
    file_name = m_state_file_name(pd=pd, S=S)
    return get_attractor(file_name)

def ds_to_np(ds):
    "Converts ds point to np array."
    X = ds.X.values
    T = ds.T.values
    return np.append(X, T)

def get_ds_points(ds, n):
    "Sample n points from a ds without replacament."
    time_points = np.random.choice(ds.time, n, replace=False)
    return [ds_to_np(ds.sel(time=x)) for x in time_points]
