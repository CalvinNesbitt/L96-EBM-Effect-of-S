"""
Contains functions that run a determinsitic simulation of the L96-EBM for specified values of S and save output.
"""
########################################################################
## Imports
########################################################################

# Standard Packages
import numpy.random as rm
import itertools
import sys
import argparse

# Custom Code Imports
from generating_attractor_data_IO import attractor_folder_name
from l96EBM import *

########################################################################
## Function for generating attractor data
########################################################################

def generate_attractor_data(S, attractor, dt=0.1, N=10000, save_dir=None):
    """
    For provided value of S, computes sb/w attractor and saves .nc file.

    - S, float: Solar parameter in the L96-EBM.
    - attractor, string: Either 'w' or 'sb'.
    - dt, float: time between obsercaitons in attractor data.
    - N, integrer: number of observaitons in attractor data.
    - save_dir, string: Where file is saved. Defaults if set to none.
    """

    # Get Save Directory name
    if save_dir is None:
        save_dir = attractor_folder_name(attractor, S=S)

    # Ensure Directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f'Made directory at {save_dir}')

    # Get IC
    if attractor == 'sb':
        x_values = 5 * np.random.randn(50)
        ic = np.append(x_values, np.array([260]))
    elif attractor == 'w':
        x_values = 10 * np.random.randn(50)
        ic = np.append(x_values, np.array([280]))

    # Set Up Integrator and Spin Up
    p=[50, S, 0.5, 0.4, 1/180**4, 8, 270, 60, 2, 1]
    runner = Integrator(p=p)
    runner.set_state(ic)
    print('Spinning Up')
    runner.integrate(100)
    runner.time = 0

    # Observe Trajectory
    looker = TrajectoryObserver(runner)
    make_observations(runner, looker, N, dt, noprog=True)

    # Save Output
    looker.dump(save_dir, name='1')
    return

def generate_attractors(S_values, attractors, dt=0.1, N=10000):
    "Runs generate attractor for many values of S/attractor."
    setups = list(itertools.product(S_values, attractors))
    for setup in setups:
        S, attractor = setup
        print(f'**Generating {attractor} attractor data for S = {S:.3f}.**\n\n')
        generate_attractor_data(S, attractor, dt=dt, N=N)
    return

########################################################################
## Script for generating attractor data with user input
########################################################################

# Argument Parser
CLI=argparse.ArgumentParser()
CLI.add_argument("--Svalues", nargs="*", type=float, default=[10])
CLI.add_argument("--attractors", nargs="*", type=str, default=['sb'])

if __name__ == '__main__':
    args = CLI.parse_args()
    S_values = args.Svalues
    attractors = args.attractors
    generate_attractors(S_values, attractors)
