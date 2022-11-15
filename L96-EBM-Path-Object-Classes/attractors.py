"""
Base Classes for attractors in the L96-EBM.

Contents:

- L96EBMAttractor, Attractor in the L96 EBM model.
    Has functionality to plot.
"""

##########################################
## Imports
##########################################
# Standard Package Imports
import xarray as xr
import sys
import numpy as np

# Custom Imports
from plots import *

def points_within_T_slice(ds, temp_thresholds):
    ds.load()
    return ds.where(np.logical_and(ds.T>=temp_thresholds[0], ds.T<=temp_thresholds[1]), drop=True)

##########################################
## L96 EBM Path Class Definition
##########################################

class L96EBMAttractor:
    """
    Attractor in the L96 EBM model.
    """

    def __init__(self, file_location, interpolate_space=False, interpolate_time=False, state_name=None):
        """
        Parameters
        -----------
        file_location, string
            Location of the .netcdf file containing L96-EBM attractor.
        state_name, string
            Which asymptotic states the attractor is.
            One of 'm', 'sb' or 'w'.
            If not specifies attempts to determine from T values.
        """
        # Unpack path information
        self.file_location = file_location
        self.ds = xr.open_dataset(file_location)
        if interpolate_space:
            space_grid = np.arange(min(self.ds.space), max(self.ds.space) + 0.1, 0.1)
            self.ds = self.ds.interp({'space': space_grid}, method='cubic', kwargs={'bounds_error': False, 'fill_value':'extrapolate'})
        if  interpolate_time:
            dt = (self.ds.time[1] - self.ds.time[0]).item()
            self.ds = self.ds.assign_coords({'time': dt * np.arange(len(self.ds.time))})
            time_grid = np.arange(0, max(self.ds.time) + 0.01 - min(self.ds.time), 0.01)
            self.ds = self.ds.interp({'time':time_grid}, method='cubic', kwargs={'bounds_error': False, 'fill_value':'extrapolate'})

        self.attrs = self.ds.attrs
        self.S = self.attrs['S']
        self._color = None

        # Determine State name
        if state_name is None:
            self._determine_state_name()
        else:
            self.state_name = state_name

    def _determine_state_name(self):
        "Determine state name if user doesn't specify."
        if self.ds.T.values[-1] < 269:
            self.state_name = 'sb'
        elif self.ds.T.values[-1] > 271:
            self.state_name = 'w'
        else:
            self.state_name = 'm'
        return

    @property
    def X(self):
        return self.ds.X

    @property
    def T(self):
        return self.ds.T

    @property
    def color(self):
        "Default color for plotting"
        if self._color is None:
            if self.state_name == 'w':
                return 'r'
            elif self.state_name == 'sb':
                return 'b'
            elif self.state_name == 'm':
                return 'g'
        else:
            return self._color

    @color.setter
    def color(self, c):
        self._color = c

    def _observable_mean(self, obs):
        return obs(self.ds).values.flatten().mean()

    @property
    def M_mean(self):
        return self._observable_mean(momentum)

    @property
    def E_mean(self):
        return self._observable_mean(energy)

    @property
    def T_mean(self):
        return self._observable_mean(temperature)

    def clip_path(self, i, j, normalise_time = True):
        "Reduce path to just points those between time indices i and j."
        dt = (self.ds.time[1] - self.ds.time[0]).item() # assuming constant dt
        self.ds = self.ds.isel(time=slice(i, j))
        if normalise_time:
            self.ds = self.ds.assign_coords({'time': dt * np.arange(len(self.ds.time))})
        return

    @property
    def ds_as_np(self):
        "Return the data as numpy array."
        X = self.ds.X.values
        T = self.ds.T.values
        return np.column_stack((X, T))

    def get_observable_points_as_np(self, obs):
        return obs(self.ds).values.flatten()

    def get_observable_points_in_T_slice(self, obs, temp_thresholds):
        return obs(points_within_T_slice(self.ds, temp_thresholds))

    def _2d_plot_projection(self, plot, *args, c=None, fax=None, **kwargs):
        "2D plot of data in a certain projection"
        if fax is None:
            fig, ax = init_2d_fax()
            fax = [fig, ax]
        else:
            fig, ax = fax
        if c is None:
            c = self.color
        plot(self.ds, *args,  fax=fax, c=c, **kwargs)
        ax.grid()
        return fig, ax

    def ET_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(ET_plot, fax=fax, *args, **kwargs)
        return

    def MT_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(MT_plot, fax=fax, *args, **kwargs)
        return

    def EM_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(EM_plot, fax=fax, *args, **kwargs)
        return

    def T_timeseries_plot(self, *args, fax=None, **kwargs):
        T_timeseries_plot(self.ds, fax=fax, *args, **kwargs)
        return

    def E_timeseries_plot(self, *args, fax=None, **kwargs):
        E_timeseries_plot(self.ds, fax=fax, *args, **kwargs)
        return

    def M_timeseries_plot(self, *args, fax=None, **kwargs):
        M_timeseries_plot(self.ds, fax=fax, *args, **kwargs)
        return

    def X_hvm_plot(self, *args, fax=None, **kwargs):

        # Initialise fax
        if fax == None:
            fax = init_2d_fax()
        fig, ax = fax

        # Reinterpolate data so plot looks better
        interpolated_ds = self.ds.interp(time=np.arange(min(self.ds.time), max(self.ds.time), 0.1), space=np.arange(1, 50, 0.1))
        interpolated_ds.X.plot(*args, ax=ax, **kwargs)
        return fig, ax

class L96EBMAttractorCollection:
    "Collection of different L96-EBM Attractors."

    def __init__(self, file_locations, state_name=None):

        self._attractor_list = [L96EBMAttractor(f, state_name) for f in file_locations]
        self.color = self._attractor_list[0].color

    def __getitem__(self, idx):
        return self._attractor_list[idx]

    def __len__(self):
        return len(self._attractor_list)

    def _2d_plot_projection(self, plot, *args, c=None, fax=None, **kwargs):
        "2D plot of data in a certain projection"
        if fax is None:
            fig, ax = init_2d_fax()
            fax = [fig, ax]
        else:
            fig, ax = fax
        if c is None:
            c = self.color

        for attractor in self._attractor_list:
            plot(attractor.ds, *args,  fax=fax, c=c, **kwargs)
        ax.grid()
        return fig, ax

    def ET_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(ET_plot, fax=fax, *args, **kwargs)
        return

    def MT_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(MT_plot, fax=fax, *args, **kwargs)
        return

    def EM_plot(self, *args, fax=None, **kwargs):
        self._2d_plot_projection(EM_plot, fax=fax, *args, **kwargs)
        return
