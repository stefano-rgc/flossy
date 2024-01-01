#!/usr/bin/env python3

import sys
from numba import jit, typed, float64, prange
from numba import types as nb_types
import matplotlib.transforms as tx
from matplotlib.ticker import MaxNLocator, LinearLocator
from matplotlib import widgets
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from types import SimpleNamespace
from enum import Enum
from dataclasses import dataclass, field
import copy, re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from astropy import units as u
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextBrowser, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt

def period_for_dP_plot(periods, mode='middle'): 
    """Return the array of periods with one less element plot the period spacings. 

    Args:
        periods : List[float]
            Periods array
        mode : str, optional (default='middle')
            Whether to return the middle or the edges of the period spacings.
            Defaults to 'middle'. Must be 'middle', 'left' or 'right'.

    Raises:
        ValueError: If `mode` is not one of the following values: "middle", "right", "left".

    Returns:
        List[float]:
            An array of periods with one less element than the input array.
    """
    if mode == 'middle':
        return (periods[1:]+periods[:-1])/2.
    elif mode == 'right':
        return periods[1:]
    elif mode == 'left':
        return periods[:-1]
    else:
        raise ValueError(f'`mode` is: {mode}. It has to be one of the following values: "middle", "right", "left".')

def clear_text(text): 
    """Clear the text of a matplotlib text object"""
    text.set_text('')

@jit(nopython=True)
def pattern_period(P0, dP0, Sigma, nr=5, nl=5): 
    '''
    Purpose:
        Return a sequence of periods following the parametrization of equation
        8 in Li Gang 2018 (https://doi.org/10.1093/mnras/sty2743).

    Args:
        P0 : float
            Position of the central peak

        dP0 : float
            Period spacing scale

        Sigma : float
            Rate of change of the period spacing

        nr : int, optional (default=5)
            Number of periods to the right of `P0`, including `P0`.

        nl : int, optional (default=5)
            Number of periods to the left of `P0`, not including `P0`.

    Returns:
        np.ndarray
            An array with the sequences of periods and period spacings.
    '''
    indices = np.arange(-nl, nr)
    if Sigma == 0:
        P = dP0*indices + P0
        dP = np.repeat(dP0, indices.size)
        return np.vstack((P, dP))
    else:
        P = dP0 * ((1+Sigma)**indices-1)/Sigma + P0
        dP = dP0 * (1+Sigma)**indices
        return np.vstack((P, dP))

def convert_resolution(x, dx): 
    """Convert frequency resolution to period resolution and vice versa"""
    y = 1/x
    dy = dx*y**2
    return y, dy

def get_period_resolution(p, df):
    """Compute the period resolution from the frequency resolution at a given period"""
    f = 1/p
    dp = df/f**2
    return dp

@jit(nopython=True)
def compute_cost_function_in_the_grid(
    M,
    grids,
    observations,
    nr,
    nl
):
    """
    Populate the matrix M with values of the cost function S. The matrix M is a 
    3D one whose axis are the three parameters (P0, dP0, Sigma) of the Period
    Spacing Pattern (PSP). Fixed values during the computation are observations
    `observations` and the number of periods to the right and left of the PSP
    centralpeak P0, `nr` and `nl`.

    Args:
        M (numpy.array):
            An empty 3D matrix whose axis'lengths match the lengths of each
            grid's key.
        grids (numba.typed.typeddict.Dict):
            Numba dictionary with the 3 keys: 'P0', 'dP0', 'Sigma'. Each key's
            value is a 1D numpy.array with the grid values for the corresponding
            parameter.
        observations (numba.typed.typeddict.Dict):
            Numba dictionary with the 3 keys: 'P', 'e_P', 'w' for the observed
            periods, their uncertainties and the weights, respectively. Each key
            is a 1D numpy.array.
        nr (int):
            Number of periods to the right of the PSP central peak P0.
        nl (int):
            Number of periods to the left of the PSP central peak P0.
    """
    for i in prange(grids['P0'].size):
        for j in prange(grids['dP0'].size):
            for k in prange(grids['Sigma'].size):
                M[i, j, k] = cost_function(
                    grids['P0'][i],
                    grids['dP0'][j],
                    grids['Sigma'][k],
                    nr,
                    nl,
                    observations['P'],
                    observations['w'],
                    observations['e_P']
                )

@jit(nopython=True)
def cost_function(
    P0,
    dP0,
    slope,
    nr,
    nl,
    periods,
    weight=None,
    sigma=None
):
    """
    Calculate the cost function for a given set of observations and a given
    size of the Period Spacing Pattern (PSP), i.e. the number of periods to the
    right and left of the PSP central peak P0.
    
    This cost function is defined in equation (11) in Garcia et al. 2022
    (https://doi.org/10.1051/0004-6361/202141926).

    Args:
        P0 (float):
            Central period of the PSP.
        dP0 (float):
            Period spacing at the central period of the PSP.
        slope (float):
            Rate of change of the period spacing.
        nr (int):
            Number of periods to the right of P0.
        nl (int):
            Number of periods to the left of P0.
        periods (np.array[float]):
            Observed periods
        weight (np.array[float], optional):
            Weights associated to each period. Defaults to None.
        sigma (np.array[float], optional):
            Uncertainties associated to each period. Defaults to None.

    Returns:
        S (float):
            The calculated cost function.
    """
    # Compute the period spacing pattern (PSP)
    P_model, dP_model = pattern_period(P0, dP0, slope, nr=nr, nl=nl)

    if weight is None:
        weight = np.repeat(1., periods.size)
    if sigma is None:
        sigma = np.repeat(0., periods.size)
    S = 0
    # Iterate over the terms of the sum
    for p_i, w_i, sigma_i in zip(periods, weight, sigma):
        # Find the index in the PSP periods closest to the observed period
        i = np.abs(p_i-P_model).argmin()
        p_model = P_model[i]
        dp_model = dP_model[i]
        # Compute the cost function
        S += w_i*(p_i-p_model)**2/(dp_model**2+sigma_i**2)
    return S

@dataclass(kw_only=True, slots=True)
class LinearPSP: 
    """Linear PSP and its parameters"""
    P0: float
    dP0: float
    Sigma: float = 0.0
    nr: int = 5
    nl: int = 5
    p: list[float] = field(init=False)
    dp: list[float] = field(init=False)

    def __post_init__(self) -> None:
        """Gererate PSP automatically after initialization"""
        self.generatePSP()

    def generatePSP(self) -> None:
        """Generate PSP (aka comb). I.e., Set attributes for periods and period spacings"""
        self.p, self.dp = pattern_period(self.P0, self.dP0, self.Sigma, self.nr, self.nl)

class Interaction(Enum):
    """Interaction modes"""
    PICKER = 'Picker tool'
    SPANER = 'Span tool'
    NONE = 'No interaction'

class flossyGUI:
    """
    Interactive GUI to explore periods in search of a linear Period Spacing
    Patterns (PSP).

    For more information, see github.com/andrespp/flossy, the help button in the
    GUI or the individual methods' docstrings. # Todo: update link to github
    """

    def __init__(
        self,
        pw_periods,
        pw_e_periods,
        pw_amplitudes,
        pg_periods=None,
        pg_amplitudes=None,
        freq_resolution=1/u.yr,
        writable_text_boxes=False,
        show_relative_error=False,
        use_bounded_fit=True,
        use_bounded_fit_P0=False,
        use_bounded_fit_dP0=False,
        use_bounded_fit_Sigma=False,
        min_dP0=100*u.s,
        max_dP0=4000*u.s,
        min_Sigma=-0.35*u.dimensionless_unscaled,
        max_Sigma=0.35*u.dimensionless_unscaled,
        decimal_digits_to_display=6,
        grid_resolution_P0=10*u.s,
        grid_resolution_dP0=1*u.s,
        grid_resolution_Sigma=1e-3*u.dimensionless_unscaled,
        grid_half_window_P0=450*u.s,
        grid_half_window_dP0=45*u.s,
        grid_half_window_Sigma=5e-2*u.dimensionless_unscaled,
        ID=None
    ): 
        """
        Initialize the GUI.

        Args:
            pw_periods (array-like with astropy quantity): 
                Prewhitening periods. Must be an array-like with astropy
                quantity convertable to day.
            pw_e_periods (array-like with astropy quantity):
                Prewhitening periods uncertanties. Must be an array-like with
                astropy quantity convertable to day.
            pw_amplitudes (array-like with astropy quantity):
                Prewhitening amplitudes. Must be an array-like with astropy
                quantity convertable to astropy dimensionless unit.
            pg_periods (array-like with astropy quantity, optional):
                Periodogram periods. Must be an array-like with astropy
                quantity convertable to day. Defaults to None.
            pg_amplitudes (array-like with astropy quantity, optional):
                Periodogram amplitudes. Must be an array-like with astropy
                quantity convertable to astropy dimensionless unit.
                Defaults to None.
            freq_resolution (astropy quantity, optional):
                Frequency resolution to indicate unresolved periods. Must be an
                astropy quantity convertable to 1/day. Defaults to 1/(u.yr).
            writable_text_boxes (bool, optional):
                Display interactive text boxes where the user can manually set
                the value of the PSP parameters. The use of writable text boxes
                slows down the GUI. Therefore, they are disabled by default
                (more information on StackOverflow:
                https://stackoverflow.com/a/59613370/9290590).
                Defaults to False.
            show_relative_error (bool, optional):
                Show relative error for the PSP parameters. The cost function is
                not analytical and very non-linear. Therefore, the estimate of
                the Hessian matrix may be not accurate and hence the errors.
                Defaults to False.
            use_bounded_fit (bool, optional):
                Use bounds for all the parameters during the fit of the PSP.
                Defaults to True.
            use_bounded_fit_P0 (bool, optional):
                Use bounds for the parameter P0 during the fit of the PSP. If
                using this option, then set `use_bounded_fit=False`.
                Defaults to False.
            use_bounded_fit_dP0 (bool, optional):
                Use bounds for the parameter dP0 during the fit of the PSP. If
                using this option, then set `use_bounded_fit=False`.
                Defaults to False.
            use_bounded_fit_Sigma (bool, optional):
                Use bounds for the parameter Sigma during the fit of the PSP. If
                using this option, then set `use_bounded_fit=False`.
                Defaults to False.
            min_dP0 (astropy quantity, optional):
                Minimum value for the dP0 slider as well as the value to clip the
                boundaries of the fit if the user tries to fit below this value
                (assuming that the fit is bounded). Must be an astropy quantity
                convertable to day. Defaults to 100*u.s.
            max_dP0 (astropy quantity, optional):
                Maximum value for the dP0 slider as well as the value to clip the
                boundaries of the fit if the user tries to fit above this value
                (assuming that the fit is bounded). Must be an astropy quantity
                convertable to day. Defaults to 4000*u.s.
            min_Sigma (astropy quantity, optional):
                Minimum value for the Sigma slider as well as the value to clip the
                boundaries of the fit if the user tries to fit below this value
                (assuming that the fit is bounded). Must be an astropy quantity
                convertable to 1. Defaults to -0.35*u.dimensionless_unscaled.
            max_Sigma (astropy quantity, optional):
                Maximum value for the Sigma slider as well as the value to clip the
                boundaries of the fit if the user tries to fit above this value
                (assuming that the fit is bounded). Must be an astropy quantity
                convertable to 1. Defaults to 0.35*u.dimensionless_unscaled.
            decimal_digits_to_display (int, optional):
                Number of decimal digits to display on the GUI. Defaults to 6.
            grid_resolution_P0 (astropy quantity, optional):
                Resolution of the grid on the `P0` axis used for the cost
                function 1D and 2D exploratory plots and the P0 slider.
                Must be an astropy quantity convertable to day.
                Defaults to 10*u.s.
            grid_resolution_dP0 (astropy quantity, optional): 
                Resolution of the grid on the `dP0` axis used for the cost
                function 1D and 2D exploratory plots and the dP0 slider.
                Must be an astropy quantity convertable to day.
                Defaults to 1*u.s.
            grid_resolution_Sigma (astropy quantity, optional):
                Resolution of the grid on the `Sigma` axis used for the cost
                function 1D and 2D exploratory plots and the Sigma slider.
                Must be an astropy quantity convertable to 1.
                Defaults to 1e-3*u.dimensionless_unscaled.
            grid_half_window_P0 (astropy quantity, optional):
                Half of the window's size on the P0 axis used for the 1D and 2D
                exploratory plots. Must be an astropy quantity convertable to day.
                Defaults to 450*u.s.
            grid_half_window_dP0 (astropy quantity, optional):
                Half of the window's size on the dP0 axis used for the 1D and 2D
                exploratory plots. Must be an astropy quantity convertable to day.
                Defaults to 45*u.s.
            grid_half_window_Sigma (astropy quantity, optional):
                Half of the window's size on the Sigma axis used for the 1D and 2D
                exploratory plots. Must be an astropy quantity convertable to 1.
                Defaults to 5e-2*u.dimensionless_unscaled.
            ID (None, optional):
                ID or title to display on the GUI and use as tentative filename
                for the output files. Defaults to None.
        
        Returns:
            None
        """

        def set_attributes_without_units():
            """Set attributes without units."""
            self.writable_text_boxes = writable_text_boxes
            self.show_relative_error = show_relative_error
            self.use_bounded_fit = use_bounded_fit
            self.use_bounded_fit_P0 = use_bounded_fit_P0
            self.use_bounded_fit_dP0 = use_bounded_fit_dP0
            self.use_bounded_fit_Sigma = use_bounded_fit_Sigma
            self.decimal_digits_to_display = decimal_digits_to_display
            self.ID = ID

        def set_attributes_with_units():
            """Convert units to the target units, strip them from units, and set
            them as attributes."""
            valid_types = (
                u.Unit,
                u.core.IrreducibleUnit,
                u.core.CompositeUnit,
                u.quantity.Quantity
            )
            attribute_value_unit = {
                'pw_periods': (pw_periods,'d'),
                'pw_e_periods': (pw_e_periods,'d'),
                'pw_amplitudes': (pw_amplitudes,'1'),
                'freq_resolution': (freq_resolution,'1/d'),
                'grid_resolution_P0': (grid_resolution_P0,'d'),
                'grid_resolution_dP0': (grid_resolution_dP0,'d'),
                'grid_resolution_Sigma': (grid_resolution_Sigma,'1'),
                'grid_half_window_P0': (grid_half_window_P0,'d'),
                'grid_half_window_dP0': (grid_half_window_dP0,'d'),
                'grid_half_window_Sigma': (grid_half_window_Sigma,'1'),
                'min_dP0': (min_dP0,'d'),
                'max_dP0': (max_dP0,'d'),
                'min_Sigma': (min_Sigma,'1'),
                'max_Sigma': (max_Sigma,'1')
            }
            if pg_periods is not None and pg_amplitudes is not None:
                attribute_value_unit.update({
                    'pg_periods': (pg_periods,'d'),
                    'pg_amplitudes': (pg_amplitudes,'1')
                })
            for attr, (val, unit) in attribute_value_unit.items():
                msg = f'`{attr}` must be an astropy quantity convertible to `{unit}`.'
                if not isinstance(val, valid_types):
                    msg += f' Got a {type(val)}.'
                    raise TypeError(msg)
                try:
                    setattr(self, attr, val.to(unit).value)
                except u.UnitConversionError as e:
                    msg += f' Got units of {val.unit}.'
                    raise u.UnitConversionError(msg) from e

        def set_prewhitened_dataframe():
            """Set the prewhitened data attribute as a DataFrame sorted by period.
            Delete redunant attributes."""
            # Create the dataframe
            pw = pd.DataFrame({
                'period': self.pw_periods,
                'e_period': self.pw_e_periods,
                'ampl': self.pw_amplitudes
            })
            # Add tag for posterior interactive use
            pw['selection'] = 1
            # Sort by period
            pw.sort_values(by='period', inplace=True)
            pw.reset_index(drop=True, inplace=True)
            self.pw = pw
            # Delete redundant attributes
            del self.pw_periods
            del self.pw_e_periods
            del self.pw_amplitudes

        def set_periodogram_dataframe():
            """Set the periodogram data attribute as a DataFrame sorted by period.
            Delete redunant attributes."""
            # Create the dataframe
            pg = pd.DataFrame({
                'period': self.pg_periods,
                'ampl': self.pg_amplitudes
            })
            # Sort by period
            pg.sort_values(by='period', inplace=True)
            pg.reset_index(drop=True, inplace=True)
            self.pg = pg
            # Delete redundant attributes
            del self.pg_periods
            del self.pg_amplitudes

        def set_grid_resolution():
            """Set attribute containing grid resolution for P0, dP0 and Sigma.
            These resolutions are used to set the sliders' step size as well as to
            set the resolution used in the exploratory plots (see button `Explore`).
            Delete redundant attributes."""
            self.grid_resolution = SimpleNamespace()
            self.grid_resolution.P0 = self.grid_resolution_P0
            self.grid_resolution.dP0 = self.grid_resolution_dP0
            self.grid_resolution.Sigma = self.grid_resolution_Sigma
            # Delete redundant attributes
            del self.grid_resolution_P0
            del self.grid_resolution_dP0
            del self.grid_resolution_Sigma
            
        def set_grid_half_window():
            """Set attribute containing grid half window for P0, dP0 and Sigma.
            These half windows are used to set the half window used in the
            exploratory plots (see button `Explore`). Delete redundant attributes."""
            self.grid_half_window = SimpleNamespace()
            self.grid_half_window.P0 = self.grid_half_window_P0
            self.grid_half_window.dP0 = self.grid_half_window_dP0
            self.grid_half_window.Sigma = self.grid_half_window_Sigma
            # Delete redundant attributes
            del self.grid_half_window_P0
            del self.grid_half_window_dP0
            del self.grid_half_window_Sigma
                            
        def set_parameter_ranges():
            """Set attribute containing parameter ranges for P0, dP0 and Sigma.
            These ranges are used to set the sliders' limits as well as to clip the
            boundaries of the fit if the user tries to fit outside of these ranges, 
            assuming that the fit is bounded (see variable `use_bounded_fit`).
            Delete redundant attributes."""
            self.parameter_ranges = SimpleNamespace()
            # P0's range
            self.parameter_ranges.P0 = (
                self.pw.period.min(),
                self.pw.period.max()
            )
            # dP0's range
            self.parameter_ranges.dP0 = (
                self.min_dP0,
                self.max_dP0
            )
            # Sigma's range
            self.parameter_ranges.Sigma = (
                self.min_Sigma,
                self.max_Sigma
            )
            # Delete redundant attributes
            del self.min_dP0
            del self.max_dP0
            del self.min_Sigma
            del self.max_Sigma
        # Handle user input
        set_attributes_without_units()
        set_attributes_with_units()        
        set_prewhitened_dataframe()
        if pg_periods is not None and pg_amplitudes is not None:
            set_periodogram_dataframe()
        set_parameter_ranges()
        set_grid_resolution()
        set_grid_half_window()
        # Create container for plot elements (e.g. lines, patches, etc)
        self.set_plot_namespace()
        # Estimate first linear PSP or comb
        self.guess_psp()
        # Estimate value for the echelle dp (so to plot later p % dp)
        self.echelle_dp = self.PSP.dP0
        # Interavtive variables
        self.initialize_interactive_variables()
        # Creates the GUI
        self.create_main_GUI()
        self.format_main_GUI()
        self.hSpanSelector(self.axs.pg)
        self.vSpanSelector(self.axs.dp)
        # Make plots
        plot_pg = True if pg_periods is not None and pg_amplitudes is not None else False
        self.plot_data(plot_pg=plot_pg)
        self.plot_echelle_dP()
        self.plot_PSP()
        self.plot_resolution_in_dp(self.freq_resolution)
        self.tweak_pg_xlim()

    def initialize_interactive_variables(self):
        """Initialize interactive variables"""
        # colors for observed periods fitted and not fitted
        self.color_period_on_off = pd.Series(data=['lightgrey', 'black'], index=[0, 1])
        # colors for the GUI buttons
        self.button_colors = {'on': '0.95', 'off': '0.85'}
        # container for the PSP previous to the press of the fit button
        self.previous_PSP = None 
        # boolean to check if the fit comes from the fit button
        self.from_fit_button = False 
        # Information about the fit
        self.fit = None 
        # contains the fit residuals
        self.textBox_residuals = None 
        # contains the relative errors for the parameters of the fit
        self.relative_err = None 
        # contains interaction mode (either PICKER, SPANER, NONE)
        self.interaction = Interaction.NONE 
        # contains linestyle of the period spacing plot
        self.linestyle_dp = '--' 
        # boolean to check if the period spacing resolution is shown
        self.show_dp_resolution = True 
        # contains mode in which the period spacings are plotted (either 'left' or 'middle')
        self.pg_mode = 'left'
        # Contains the limits of the windows in which the amplitude slider acts
        self.window_pmin = self.pw.period.min()
        self.window_pmax = self.pw.period.max()
        # contains the widget matplotlib connection ids
        self.connections_main_GUI = []
        self.connections_secondary_GUI = []
        # contains the Matplot figure objects
        self.main_fig =  None
        self.secondary_fig = None

    def tweak_pg_xlim(self): 
        """Tweak xlim for a better visualization"""
        p = self.pw.period.values
        xrange = p.max()-p.min()
        xrange *= 1.2
        xmid = (p.max()+p.min())/2
        self.axs.pg.set_xlim(
            xmid-xrange/2,
            xmid+xrange/2
        )

    def set_plot_namespace(self):
        """Set a namespace as attribute to store plot contents such as lines,
        scatter, etc."""
        self.plots = SimpleNamespace(
            echelle_vline1=None, # one of the two vertical lines in the echelle plot
            echelle_vline2=None, # one of the two vertical lines in the echelle plot
            p_scatter_0=None, # triangles above the PG plot denoting selected periods
            p_scatter_1=None, # a third of circles in the echelle plot for the observed periods
            p_scatter_2=None, # a third of circles in the echelle plot for the observed periods
            p_scatter_3=None, # a third of circles in the echelle plot for the observed periods
            p_lines=None, # dotted vertical lines in PG plot denoting observed periods used during the fit
            dp_lines=None, # line connecting the data in the period spacing plot
            dp_resolution=None, # line showing the resolution of the period spacing plot
            PSP_pg_lines1=None, # vertical lines of the PSP periods
            PSP_pg_lines2=None, # vertical lines of the unresolved PSP periods
            PSP_pg_vline=None, # vertical line for P0
            PSP_dp_lines=None, # line connecting the period spacings of the PSP
            PSP_dp_dot=None, # gold/yellow star that marks location of P0 in the period spacing plot
            PSP_echelle_scatter_1=None, # a third of the circles of the echelle diagram for the PSP periods
            PSP_echelle_scatter_2=None, # a third of the circles of the echelle diagram for the PSP periods
            PSP_echelle_scatter_3=None, # a third of the circles of the echelle diagram for the PSP periods
            PSP_echelle_dot_1=None, # a third of the circles of the echelle diagram for the P0
            PSP_echelle_dot_2=None, # a third of the circles of the echelle diagram for the P0
            PSP_echelle_dot_3=None, # a third of the circles of the echelle diagram for the P0
            dp_hline=None, # horizontal line from period spacing plot that marks period spacing used for echelle diagram
            matches_scatter=None, # green triangles indicating which observations are consistent with the PSP
            P0dP0_cbar=None, # color bar for the P0dP0 plot
            dP0Sigma_cbar=None, # color bar for the dP0Sigma plot
            SigmaP0_cbar=None # color bar for the SigmaP0 plot
        )

    def __enter__(self):
        """Establish a connection and return the object of the same class"""
        self.enable_mpl_connections_main_GUI()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Disconnect the connection """
        self.disconnect()

    def enable_mpl_connections_main_GUI(self):
        """Install the event handlers for the plot."""
        self.connections_main_GUI = [
            self.main_fig.canvas.mpl_connect('button_press_event', self.on_click_main_figure),
            self.main_fig.canvas.mpl_connect('pick_event', self.on_pick)
        ]

    def enable_mpl_connections_secondary_GUI(self):
        """Install the event handlers for the secondary plot."""
        # Disconect previous connections
        if len(self.connections_secondary_GUI) > 0:
            for connection in self.connections_secondary_GUI:
                self.secondary_fig.canvas.mpl_disconnect(connection)
        # Connect new connections
        self.connections_secondary_GUI = [
            self.secondary_fig.canvas.mpl_connect('button_press_event', self.on_click_secondary_figure)
        ]

    def disconnect(self):
        """Uninstall the event handlers for the plot."""
        for connection in self.connections_main_GUI+self.connections_secondary_GUI:
            self.main_fig.canvas.mpl_disconnect(connection)

    def set_interaction_mode(self,mode):
        """Sets the interaction mode to either 'picker' or 'spanner' or 'none'."""
        if mode == 'picker':
            self.interaction = Interaction.PICKER
            text = self.buttons.clicking.label.get_text()
            self.buttons.clicking.color = self.button_colors['on']
            self.buttons.clicking.label.set_text(text.replace('OFF','ON'))
            text = self.buttons.spanning.label.get_text()
            self.buttons.spanning.color = self.button_colors['off']
            self.buttons.spanning.label.set_text(text.replace('ON','OFF'))
        elif mode == 'spanner':
            self.interaction = Interaction.SPANER
            text = self.buttons.clicking.label.get_text()
            self.buttons.clicking.color = self.button_colors['off']
            self.buttons.clicking.label.set_text(text.replace('ON','OFF'))
            text = self.buttons.spanning.label.get_text()
            self.buttons.spanning.color = self.button_colors['on']
            self.buttons.spanning.label.set_text(text.replace('OFF','ON'))
        elif mode == 'none':
            self.interaction = Interaction.NONE
            text = self.buttons.clicking.label.get_text()
            self.buttons.clicking.color = self.button_colors['off']
            self.buttons.clicking.label.set_text(text.replace('ON','OFF'))
            text = self.buttons.spanning.label.get_text()
            self.buttons.spanning.color = self.button_colors['off']
            self.buttons.spanning.label.set_text(text.replace('ON','OFF'))
        else:
            raise ValueError('Invalid interaction mode.')
        self.main_fig.canvas.draw_idle()

    def on_click_secondary_figure(self, event):
        """Handle click events on different axes of the secondary figure"""

        def on_dP0_Sigma_axes(): 
            """Set parameters dP0 and Sigma"""
            if event.button == 1:
                self.sliders.dP0.set_val(event.xdata)
                self.sliders.Sigma.set_val(event.ydata)
        
        def on_P0_dP0_axes(): 
            """Set parameters P0 and dP0"""
            if event.button == 1:
                self.sliders.P0.set_val(event.xdata)
                self.sliders.dP0.set_val(event.ydata)
        
        def on_Sigma_P0_axes(): 
            """Set parameters Sigma and P0"""
            if event.button == 1:
                self.sliders.Sigma.set_val(event.xdata)
                self.sliders.P0.set_val(event.ydata)
        
        def on_P0_axes(): 
            """Set parameter P0"""
            if event.button == 1:
                self.sliders.P0.set_val(event.xdata)
        
        def on_dP0_axes(): 
            """Set parameter dP0"""
            if event.button == 1:
                self.sliders.dP0.set_val(event.xdata)
        
        def on_Sigma_axes(): 
            """Set parameter Sigma"""
            if event.button == 1:
                self.sliders.Sigma.set_val(event.xdata)

        # Check for the correct interaction mode
        if not self.interaction == Interaction.PICKER:
            return
        # Check for the correct axes
        ax = event.inaxes
        axs = self.axs
        if ax == axs.dP0Sigma:
            on_dP0_Sigma_axes()
        elif ax == axs.P0dP0:
            on_P0_dP0_axes()
        elif ax == axs.SigmaP0:
            on_Sigma_P0_axes()
        elif ax == axs.P0:
            on_P0_axes()
        elif ax == axs.dP0:
            on_dP0_axes()
        elif ax == axs.Sigma:
            on_Sigma_axes()
        else:
            return
        # Clear fit information
        self.clear_fit_results()
        
    def on_click_main_figure(self, event):
        """Handle click events on different axes of the main figure"""

        def on_periodogram_axes(): 
            """Add/remove lines to/from the PSP (aka comb). Update PSP and its plots"""
            nl = self.PSP.nl # Number of lines to the left of P0
            nr = self.PSP.nr # Number of lines to the right of P0
            # Add 1 line
            if event.button == 1:
                if event.xdata < self.PSP.P0:
                    nl += 1
                else:
                    nr += 1
            # Remove 1 line
            if event.button == 3:
                if event.xdata < self.PSP.P0:
                    nl -= 1
                else:
                    nr -= 1
            # Add/Remove 5 lines
            if event.button == 2:
                if event.xdata < self.PSP.P0:
                    nl += 5 if event.xdata < self.PSP.p[0] else -5
                else:
                    nr += 5 if event.xdata > self.PSP.p[-1] else -5
            # Ensure acceptable value
            self.PSP.nl = max(nl, 0)
            self.PSP.nr = max(nr, 2)
            # Generate and plot linear PSP (aka comb)
            self.PSP.generatePSP()
            self.plot_PSP()

        def on_period_spacing_axes(): 
            """Set parameters P0 and dP0 or Sigma"""
            # Read input
            if self.pg_mode == 'middle':
                P0 = event.xdata - event.ydata/2
            elif self.pg_mode == 'left':
                P0 = event.xdata
            else:
                raise ValueError(f"pg_mode must be 'middle' or 'left', not {self.pg_mode}")
            dP0 = event.ydata
            # Set P0 and dP0
            if event.button == 1:
                self.sliders.P0.set_val(P0)
                self.sliders.dP0.set_val(dP0)
            # Set Sigma
            if event.button == 3:
                if self.PSP.P0 != P0:
                    Sigma = (self.PSP.dP0-dP0) / (self.PSP.P0-P0)  # slope formula
                    self.sliders.Sigma.set_val(Sigma)
        
        # Check for the correct interaction mode
        if not self.interaction == Interaction.PICKER:
            return
        # Check for the correct axes
        ax = event.inaxes
        axs = self.axs
        if ax == axs.pg:
            on_periodogram_axes()
        elif ax == axs.dp:
            on_period_spacing_axes()
        else:
            return
        # Clear fit information
        self.clear_fit_results()
        
    def on_pick(self, event): 
        """Handle pick events on different axes. It checks for the correct
        interaction mode and axes, and if the correct conditions are met, it
        swaps the selected period on and off of the fit. It also updates the
        plots."""
        # Check for the correct interaction mode
        if not self.interaction == Interaction.PICKER:
            return
        pw = self.pw
        # Check for the correct axes
        if event.mouseevent.inaxes == self.axs.p:
            i = (pw.period-event.mouseevent.xdata).abs().argmin()
        elif event.mouseevent.inaxes == self.axs.echelle:
            i = (pw.period-event.mouseevent.ydata).abs().argmin()
        else:
            return
        # Swap picked period
        swap = {0: 1, 1: 0}
        pw.loc[i, 'selection'] = swap[pw.loc[i, 'selection']]
        # Update data plots
        self.plot_data(echelle_keep_xlim_ylim=True)
        self.clear_fit_results()
        # Update the figure
        self.main_fig.canvas.draw_idle()

    def on_hspan(self, vmin, vmax): 
        """Handles the horizontal span evente on the periodogram axes. It checks
        for the correct interaction mode, and if the correct conditions are met,
        it updates the period selection for the fit and the range of the P0
        slider. It also updates the plots."""

        # Check for the correct interaction mode
        if not self.interaction == Interaction.SPANER:
            return
        if vmin == vmax:
            return
        # Get the periods within the selected range
        pw = self.pw.query('period>@vmin and period<@vmax')
        if not pw.period.size > 0:
            return
        # Update the period window for the amplitude slider usage
        self.window_pmin = vmin 
        self.window_pmax = vmax
        # Update period selection to the selected range
        i = (self.pw.period > vmin) & (self.pw.period < vmax)
        self.pw.selection = 0
        self.pw.loc[i, 'selection'] = 1
        # Update data plots
        xlim = self.axs.pg.get_xlim()
        self.plot_data(echelle_keep_xlim_ylim=True)
        self.axs.pg.set_xlim(xlim)
        # Update range of slider P0
        ax = self.axs.sliders.P0
        slider = self.sliders.P0
        valinit = slider.val
        self.update_slider(ax, slider, vmin, vmax, valinit)

    def on_vspan(self, vmin, vmax): 
        """Handles the vertical span evente on the period spacing axes. It checks
        for the correct interaction mode, and if the correct conditions are met,
        it updates the range of the dP0 slider. It also updates the plots."""

        # Check for the correct interaction mode
        if not self.interaction == Interaction.SPANER:
            return
        # Ensure positive values
        vmin = max(vmin, 0)
        vmax = max(vmax, 0)
        # Ensure distinct values 
        if vmin == vmax:
            return
        # Update slider dP0
        ax = self.axs.sliders.dP0
        slider = self.sliders.dP0
        valinit = slider.val
        self.update_slider(ax, slider, vmin, vmax, valinit)
        # update slider echelle_dP
        ax = self.axs.sliders.echelle_dP
        slider = self.sliders.echelle_dP
        valinit = slider.val
        self.update_slider(ax, slider, vmin, vmax, valinit)

    def hSpanSelector(self, ax): 
        """Creates a horizontal span selector widget on the given axes."""
        prop = dict(facecolor='grey', alpha=0.20)
        self.hspan = widgets.SpanSelector(ax, self.on_hspan, 'horizontal', props=prop, useblit=False)

    def vSpanSelector(self, ax):
        """Creates a vertical span selector widget on the given axes."""
        prop = dict(facecolor='grey', alpha=0.20)
        self.vpan = widgets.SpanSelector(ax, self.on_vspan,  'vertical', props=prop, useblit=False)

    def toggle_clicking_mode(self, event):
        """Toggles the clicking mode on and off."""
        if self.interaction != Interaction.PICKER:
            self.set_interaction_mode('picker')            
        else:
            self.set_interaction_mode('none')

    def toggle_spanning_mode(self, event):
        """Toggles the spanning mode on and off."""
        if self.interaction != Interaction.SPANER:
            self.set_interaction_mode('spanner')
        else:
            self.set_interaction_mode('none')            

    def toggle_dp_line_style(self, event):
        """Toggle the line style of the period spacing plot and update the
        button label acordingly."""
        text = self.buttons.dp_style.label.get_text()
        if 'HIDE' in text:
            ls = 'None'
            text = text.replace('HIDE','SHOW')
        elif 'SHOW' in text:
            ls = '--'
            text = text.replace('SHOW','HIDE')
        else:
            raise ValueError('Button label must contain "HIDE" or "SHOW".')
        self.linestyle_dp = ls
        self.plots.dp_lines[0].set_linestyle(ls)
        self.buttons.dp_style.label.set_text(text)
        self.main_fig.canvas.draw_idle()

    def toggle_dp_resolution_line(self, event):
        """Toggle the visibility of the resolution line in the period spacing
        plot and update the button label acordingly."""
        text = self.buttons.dp_resolution.label.get_text()
        if 'HIDE' in text:
            self.show_dp_resolution = False
            self.plots.dp_resolution[0].remove()
            text = text.replace('HIDE','SHOW')
        elif 'SHOW' in text:
            self.show_dp_resolution = True
            self.plot_resolution_in_dp(self.freq_resolution)
            text = text.replace('SHOW','HIDE')
        else:
            raise ValueError('Button label must contain "HIDE" or "SHOW".')
        self.axs.dp.legend().update({})
        self.buttons.dp_resolution.label.set_text(text)
        self.main_fig.canvas.draw_idle()

    def populate_help_content(self, text_browser):
        # You can define your help content here. For example:
        help_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>Help</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: lightgray;
            line-height: 1.6;
        }
        .head {
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            border-bottom: 2px solid black;
            padding-bottom: 5px;
        }
        .bold {
            font-size: 18px;
            font-weight: bold;
        }
        .underline {
            font-size: 18px;
            text-decoration: underline;
        }
        .monospaced, .monospaced_underline {
            font-family: 'Courier New', monospace;
            font-size: 18px;
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            padding: 2px 4px;
            border-radius: 4px;
        }
        .monospaced_underline {
            text-decoration: underline;
        }
        .bullet1, .bullet1bold, .bullet2, .bullet2bold, .bullet3 {
            font-size: 18px;
            margin-left: 20px;
        }
        .bullet1bold, .bullet2bold {
            font-weight: bold;
        }
        .bullet2, .bullet2bold {
            margin-left: 40px;
        }
        .bullet3 {
            margin-left: 60px;
        }
        .paper, .zenodo {
            color: blue;
            text-decoration: underline;
        }
        .code-snippet {
            background-color: #eee;
            border-left: 3px solid #f36d33;
            padding: 0.5em;
            margin: 1em 0;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="head">1. Leyend</div>
    <div class="bullet1"><span class="underline">Black triangles, circles and lines</span>: Periods considered for the fit.</div>
    <div class="bullet1"><span class="underline">Grey triangles, circles and lines</span>: Periods not considered for the fit.</div>
    <div class="bullet1"><span class="underline">Red stars, circles and lines</span>: Periods in the linear Period Spacing Pattern (PSP).</div>
    <div class="bullet1"><span class="underline">Yellow/gold circle and line</span>: Location of P0 and/or dP0.</div>
    <div class="bullet1"><span class="underline">Green triangles</span>: Denote observed periods that are consistent with fitted PSP within the frequency resolution.</div>
    <div class="bullet1"><span class="underline">Purple vertical lines</span>: Period spacing is unresolved.</div>
    <div class="bullet1"><span class="underline">S</span>: Cost function of the fit. Taken from Garcia et al. 2022 (<a href="https://doi.org/10.1051/0004-6361/202141926" class="paper">link</a>).</div>
    <div class="bullet1"><span class="underline">Residuals</span>: Vanilla expression <span class="monospaced">np.sum(np.abs(p-PSP.p).min() for p in p_obs)</span>.</div>

    <div class="head">2. Clicker mode</div>
    <div class="bullet1bold">Periodogram</div>
    <div class="bullet2"><span class="monospaced_underline">Left mouse button</span>: Adds a comb period from the right (left) when clicking to the right (left) of P0.</div>
    <div class="bullet2"><span class="monospaced_underline">Right mouse button</span>: Removes a comb period from right (left) when clicking to the right (left) of P0.</div>
    <div class="bullet2"><span class="monospaced_underline">Center mouse button</span>: Adds/removes five comb periods when clicking to the left/right of leftmost comb's period.</div>
    <div class="bullet2"><span class="monospaced_underline">Center mouse button</span>: Adds/removes five comb periods when clicking to the right/left of rightmost comb's period.</div>

    <div class="bullet1bold">Period spacing</div>
    <div class="bullet2"><span class="monospaced_underline">Left mouse button</span>: Set comb's P0 and dP0.</div>
    <div class="bullet2"><span class="monospaced_underline">Right mouse button</span>: Set comb's slope (Sigma).</div>

    <div class="bullet1bold">Echelle diagram</div>
    <div class="bullet2"><span class="monospaced_underline">Left mouse button</span>: If on a period, adds/removes it from the set of observations to be fitted to the comb.</div>

    <div class="bullet1bold">Top triangles</div>
    <div class="bullet2"><span class="monospaced_underline">Left mouse button</span>: If on a triangle, adds/removes the corresponding period from the set of observations to be fitted to the comb.</div>

    <div class="bullet1bold">Cost function</div>
    <div class="bullet2"><span class="monospaced_underline">Left mouse button</span>: Set new values for the comb's parameters.</div>

    <div class="head">3. Spanner mode</div>
    <div class="bullet1bold">Periodogram</div>
    <div class="bullet2"><span class="monospaced_underline">Horizontal span with the left mouse button</span>: Select the observations to be fitted to the comb and determine the range of the P0 slider.</div>

    <div class="bullet1bold">Period spacing</div>
    <div class="bullet2"><span class="monospaced_underline">Vertical span with the left mouse button</span>: Determine the range of the dP0 and modulo sliders.</div>

    <div class="head">4. Sliders</div>
    <div class="bullet1"><span class="underline">P0</span>: Sets the comb's P0.</div>
    <div class="bullet1"><span class="underline">dP0</span>: Sets the comb's dP0.</div>
    <div class="bullet1"><span class="underline">Sigma</span>: Sets the comb's Sigma (slope).</div>
    <div class="bullet1"><span class="underline">Echelle ΔP</span>: Sets the divisor of the x-axis of the echelle diagram.</div>
    <div class="bullet1"><span class="underline">Amplitude threshold (%)</span>: Sets the threshold for the periods' amplitude to be fitted.</div>
    <div class="bullet2">* Given a set of periods to fit, a value 0.2 will remove from the set periods with amplitude less than 20% of the maximum amplitude in the set.</div>

    <div class="head">5. Buttons</div>
    <div class="bullet1"><span class="underline">Fit</span>: Fit the PSP (comb of vertical red lines) to the selected observations (black triangles and black vertical dashed lines).</div>
    <div class="bullet2">* Note that the fit is performed on the period space and not on the period spacing space. Therefore, missing modes will not impact the fit.</div>
    <div class="bullet2">* The fit uses the following bounded parameters by default (see variable `use_bounded_fit`):</div>
    <div class="bullet3">- P0 ∈ [PSP.P0 - Δ, PSP.P0 + Δ], where Δ = PSP.dP0/2</div>
    <div class="bullet3">- dP0 ∈ [PSP.dP0/2, 2*PSP.dP0]</div>
    <div class="bullet3">- Sigma ∈ [PSP.Sigma - Δ, PSP.Sigma + Δ], where Δ = 0.15</div>
    <div class="bullet1"><span class="underline">Undo</span>: Set the PSP parameters to the values previous to the press of the Fit button.</div>
    <div class="bullet1"><span class="underline">Save</span>: Save fit results generated by the Fit button (see Saved files section below).</div>
    <div class="bullet2">* Note that fit results are deleted every time the user changes the PSP parameters or the observations to fit.</div>
    <div class="bullet1"><span class="underline">Clicker ON/OFF</span>: Enable/disable clicker mode (see Clicker mode section above).</div>
    <div class="bullet1"><span class="underline">Spanner ON/OFF</span>: Enable/disable spanner mode (see
</body>
</html>
        '''
        text_browser.setHtml(help_content)

    def help(self, event):
        """Shows a help message when the user clicks on the GUI help button.""" 
        app = QApplication([])
        main_window = QMainWindow()
        main_window.setWindowTitle("Help")
        main_window.setGeometry(200, 100, 1300, 500)
        central_widget = QWidget()
        main_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)
        self.populate_help_content(text_browser)
        main_window.show()
        app.exec_()

    def undo(self, event):
        """Set the PSP parameters to the values previous to the press of the Fit button."""
        # Check if there is something to undo
        if self.previous_PSP is None:
            return
        # Unpack
        previous_PSP = self.previous_PSP
        sliders = self.sliders
        # Reset PSP parameters
        self.PSP.nl = previous_PSP.nl
        self.PSP.nr = previous_PSP.nr
        # Slider updates below update PSP parameters, regenerate PSP and plots it
        self.update_slider_val(sliders.P0, previous_PSP.P0)
        self.update_slider_val(sliders.dP0, previous_PSP.dP0)
        self.update_slider_val(sliders.Sigma, previous_PSP.Sigma)
        # Clear fit information
        self.clear_fit_results()

    def save(self, event):
        """Save results for the last press of the Fit button."""
        if self.fit is None:
            QMessageBox.information(None, "Save", "No fit to save.")
            return
        # Prompt user for file name
        initialvalue = f'{self.ID}' if self.ID is not None else 'PSP'
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(
            None, 
            "Save File", 
            initialvalue, 
            "All Files (*);;PDF Files (*.pdf);;CSV Files (*.csv)", 
            options=options)
        if not filename:
            return
        # Check for blank input
        if re.search("^[\s]*$", filename):
            print("Invalid file name.")
            return
        # Save GUI snapshot as a PDF
        if self.secondary_fig is None or self.secondary_fig.number not in plt.get_fignums():
            self.main_fig.savefig(f'{filename}.pdf', bbox_inches='tight')
        else:
            with PdfPages(f'{filename}.pdf') as pdf:
                pdf.savefig(self.main_fig, bbox_inches='tight')
                pdf.savefig(self.secondary_fig, bbox_inches='tight')
        # Save PSP as a CSV
        observations = self.fit.fitted_pw.copy()
        observations.drop(columns=['selection'], inplace=True)
        fit = self.fit.fitted_PSP.copy()
        cols = [
            'index_PSP',
            'period_PSP',
            'period_spacing_PSP',
            'missing_mode_PSP'
        ]
        fit = fit[cols]
        df = pd.merge(
            observations,
            fit,
            how='outer',
            left_on='match_index_PSP',
            right_on='index_PSP'
        )
        df.to_csv(f'{filename}_all.csv', index=False)

    def read_box_P0(self, text):
        """Emulate a P0 slider action by reading the value of a text box if `text` is numeric."""
        try:
            self.sliders.P0.set_val(float(text))
        except ValueError:
            pass

    def read_box_dP0(self, text):
        """Emulate a dP0 slider action by reading the value of a text box if `text` is numeric."""
        try:
            self.sliders.dP0.set_val(float(text))
        except ValueError:
            pass

    def read_box_Sigma(self, text):
        """Emulate a Sigma slider action by reading the value of a text box if `text` is numeric."""
        try:
            self.sliders.Sigma.set_val(float(text))
        except ValueError:
            pass

    def explore_PSP_vecinity(self, event=None):
        """Explore the values of the cost function in the vecinity of the 
        current PSP parameters. Plot the values in a secondary GUI (figure)
        with Matplotlib connections (interactive functions). Display the GUI.
        The vecinity is defined by the variables`grid_resolution` and
        `grid_half_window`.
        """
        # Unpack
        PSP = self.PSP
        pw = self.pw
        P0_resolution = self.grid_resolution.P0
        dP0_resolution = self.grid_resolution.dP0
        Sigma_resolution = self.grid_resolution.Sigma
        P0_half_window = self.grid_half_window.P0
        dP0_half_window = self.grid_half_window.dP0
        Sigma_half_window = self.grid_half_window.Sigma
        # Grids
        grids = typed.Dict.empty(
            key_type=nb_types.unicode_type,
            value_type=float64[:]
        ) # Types for Numba
        grids['P0'] = np.arange(
            PSP.P0 - P0_half_window,
            PSP.P0 + P0_half_window,
            P0_resolution
        )
        grids['dP0'] = np.arange(
            PSP.dP0 - dP0_half_window,
            PSP.dP0 + dP0_half_window,
            dP0_resolution
        )
        grids['Sigma'] = np.arange(
            PSP.Sigma - Sigma_half_window,
            PSP.Sigma + Sigma_half_window,
            Sigma_resolution
        )
        # Initialize exploration grid
        vecinity = np.empty([
            grids['P0'].size,
            grids['dP0'].size,
            grids['Sigma'].size
        ])
        # Observation to fit
        observations = typed.Dict.empty(
            key_type=nb_types.unicode_type,
            value_type=float64[:]
        ) # Types for Numba
        observations['P'] = pw.query('selection==1').period.values
        observations['e_P'] = pw.query('selection==1').e_period.values
        observations['A'] = pw.query('selection==1').ampl.values
        observations['w'] = observations['A']/observations['A'].max()
        observations['w'] /= observations['w'].sum()  # normalize the weights
        compute_cost_function_in_the_grid(
            vecinity,
            grids,
            observations,
            PSP.nr,
            PSP.nl
        )
        self.create_secondary_GUI()
        self.plot_vecinity(vecinity, grids)
        self.enable_mpl_connections_secondary_GUI()
        self.format_secondary_GUI()
        self.secondary_fig.show()

    def plot_matches(self):
        """Plot a green triangle for the observations that match the PSP model."""
        # Clear plot element if it exists
        self.clear_pathCollection(self.plots.matches_scatter)
        matches = self.fit.fitted_pw.query('matches_PSP==1').period
        if len(matches) > 0:
            x = matches
            y = np.repeat(0.6, len(x))
            prop = dict(c='limegreen', marker=7, alpha=1, zorder=2)
            self.plots.matches_scatter = self.axs.p.scatter(x, y, **prop)
        # Reinitialize container
        else:
            self.plots.matches_scatter = None

    def fitPSP(self, event):
        """Fit the period spacing pattern (PSP) to the observations. Estimate
        errors using an approximation of the Hessian matrix. Compute residuals.
        Identify the observations that match the PSP model within the frequency
        resolution. Update the GUI."""
        # Unpack
        axs = self.axs
        sliders = self.sliders
        PSP = self.PSP
        parameter_ranges = self.parameter_ranges
        pw = self.pw
        # Update previous PSP info
        self.previous_PSP = copy.deepcopy(PSP)
        # Signal that changes are from fit button press (used later to display residuals)
        self.from_fit_button = True
        # Data to fit
        fitted_pw = pw.query('selection==1').copy()
        fitted_pw.reset_index(inplace=True, drop=True)
        fitted_pw['matches_PSP'] = 0
        periods_obs = fitted_pw.period.values
        e_periods_obs = fitted_pw.e_period.values
        amplitude_obs = fitted_pw.ampl.values
        frequency_resolution = self.freq_resolution
        # weights
        weights_obs = amplitude_obs/amplitude_obs.max()
        weights_obs /= weights_obs.sum() # normalize
        # Minimization parameters
        x0 = [PSP.P0, PSP.dP0, PSP.Sigma]
        # Bounds P0
        if self.use_bounded_fit or self.use_bounded_fit_P0:
            delta = PSP.dP0/2
            P0_bounds = (
                max(PSP.P0-delta, parameter_ranges.P0[0]),
                min(PSP.P0+delta, parameter_ranges.P0[1])
            )
        else:
            P0_bounds = (None, None)    
        # Bounds dP0
        if self.use_bounded_fit or self.use_bounded_fit_dP0:
            dP0_bounds = (
                max(0.5*PSP.dP0, parameter_ranges.dP0[0]),
                min(2.0*PSP.dP0, parameter_ranges.dP0[1])
            )
        else:
            dP0_bounds = (None, None)
        # Bounds Sigma
        if self.use_bounded_fit or self.use_bounded_fit_Sigma:
            delta = 0.15
            Sigma_bounds = (
                max(PSP.Sigma-delta, parameter_ranges.Sigma[0]),
                min(PSP.Sigma+delta, parameter_ranges.Sigma[1])
            )
        else:
            Sigma_bounds = (None, None)
        # Bounds and arguments for the fit
        bounds = [P0_bounds, dP0_bounds, Sigma_bounds]
        args = (PSP.nr, PSP.nl, periods_obs, weights_obs, e_periods_obs)
        
        def cost_function_alias(
            x,
            nr,
            nl,
            periods,
            weight,
            sigma
        ):
            '''
            Same as the function `cost_function` but collects the parameters to
            optimize in the first argument `x`. It does it to comply with the
            convention of scipy.optimize.minimize.
            '''
            return cost_function(*x, nr, nl, periods, weight, sigma)

        # Fit
        result = minimize(cost_function_alias, x0, args=args, bounds=bounds)
        # Get the approximation of the covariance matrix
        cov = result.hess_inv.todense()
        # Get the standard deviation of the parameters
        std = np.sqrt(np.diag(cov))
        # Get relative errors
        self.relative_err = SimpleNamespace(
            P0 = std[0]/result.x[0],
            dP0 = std[1]/result.x[1],
            Sigma = std[2]/result.x[2]
        )
        # Update PSP parameters
        PSP.P0 = result.x[0]
        PSP.dP0 = result.x[1]
        PSP.Sigma = result.x[2]
        PSP.generatePSP()
        # Update sliders (plot automatically updated at each line below)
        self.update_slider(
            axs.sliders.P0,
            sliders.P0,
            sliders.P0.valmin,
            sliders.P0.valmax,
            PSP.P0
        )
        self.update_slider(
            axs.sliders.dP0,
            sliders.dP0,
            sliders.dP0.valmin,
            sliders.dP0.valmax,
            PSP.dP0
        )
        self.update_slider(
            axs.sliders.Sigma,
            sliders.Sigma,
            sliders.Sigma.valmin,
            sliders.Sigma.valmax,
            PSP.Sigma
        )

        def is_within_resolution(p):
            """
            Return 0 if the observer period `p` is not consistent with any
            period in the fitted PSP within the frequency resolution. Otherwise,
            return 1.
            """
            # If `p` is iterable` return two arrays
            try:
                binary_lst = list()
                index_lst = list()
                for period in p:
                    binary, index = is_within_resolution(period)
                    binary_lst.append(binary)
                    index_lst.append(index)
                return binary_lst, index_lst
            # If `p` is not iterable, return a tuple
            except TypeError:
                # Get distance to closest period in the fitted PSP and its index
                dp = np.abs(p-PSP.p).min()
                i = np.abs(p-PSP.p).argmin()
                # Convert that distance to the frequency resolution eqivalent
                _, df = convert_resolution(p, dp)
                # Check if that distance is within the frequency resolution
                return (1,i) if df < frequency_resolution else (0,-1)
        
        # Find data that matches PSP within the data frequency resolution
        binary, index = is_within_resolution(fitted_pw.period)
        fitted_pw['matches_PSP'] = binary
        fitted_pw['match_index_PSP'] = index

        fitted_PSP = pd.DataFrame({
            'period_PSP': PSP.p,
            'period_spacing_PSP': PSP.dp,
            'index_PSP': np.arange(PSP.p.size)
        })

        def is_missing_mode(index, indexes):
            """Return 0 if `index` is in `indexes`. Otherwise, return 1."""
            return 0 if index in indexes else 1

        # Apply missing_mode function
        fitted_PSP['missing_mode_PSP'] = fitted_PSP.index_PSP.apply(
            is_missing_mode, args=(fitted_pw.match_index_PSP.values,)
        )
        # Needed for saving output to text file
        self.fit = SimpleNamespace(
            fitted_pw = fitted_pw,
            fitted_PSP = fitted_PSP
        )
        # Plot matches
        self.plot_matches()
        # Compute residuals
        residuals = sum(np.abs(p-PSP.p).min() for p in periods_obs)
        # Update boxes
        self.textBox_residuals.set_text(
            round(residuals,self.decimal_digits_to_display)
        )
        # Update signal
        self.from_fit_button = False
        self.relative_err = None

    def clear_pathCollection(self, marker):
        """Clear a matplotlib.collections.PathCollection object if it exists."""
        if marker:
            marker.remove()
            
    def clear_line2D_list(self, lines):
        """Clear a list of matplotlib.lines.Line2D objects if it exists."""
        if lines:
            for line in lines:
                line.remove()

    def plot_data(
        self,
        plot_pg=False,
        echelle_keep_xlim=False,
        echelle_keep_ylim=False,
        echelle_keep_xlim_ylim=False
    ):
        """Plot periodogram (if any), period indicators (triangles and vertical
        lines), period spacings and echelle diagram.
        
        Args:
            plot_pg (bool, optional):
                If True, plot periodogram (recommended if the periodogram is
                available). Defaults to False.
            echelle_keep_xlim (bool, optional):
                Do not update the x-axis limits of the echelle diagram.
                Defaults to False.
            echelle_keep_ylim (bool, optional):
                Do not update the y-axis limits of the echelle diagram.
                Defaults to False.
            echelle_keep_xlim_ylim (bool, optional):
                Do not update the x- and y-axis limits of the echelle diagram.
                Defaults to False.
        """
        if plot_pg:
            self.plot_pg() # periodogram
        self.plot_p_indicator() # period indicators (triangles)
        self.plot_p_as_lines_in_pg() # vertical lines in periodogram
        self.plot_dp() # period spacings
        self.plot_echelle(
            keep_xlim=echelle_keep_xlim,
            keep_ylim=echelle_keep_ylim,
            keep_xlim_ylim=echelle_keep_xlim_ylim
        ) # echelle diagram

    def plot_PSP(self):
        """Update plot of the period spacing pattern in all axes. Update button
        label if needed."""
        self.plot_PSP_in_pg() # Plot in periodogram axis
        self.plot_PSP_in_dp() # Plot in period spacings axis
        self.plot_PSP_in_echelle() # Plot in echelle diagram axis
        # Update button text if needed
        text = self.buttons.PSP.label.get_text()
        if 'SHOW' in text:
            self.buttons.PSP.label.set_text(text.replace('SHOW','HIDE'))


    def update_axes_echelle(self, **kwargs):
        """Update the echelle diagram axis."""
        
        # Get current visibility of PSP in echelle diagram
        visible_PSP = self.plots.PSP_echelle_scatter_1.get_visible()
        
        self.plot_echelle(**kwargs)
        self.plot_PSP_in_echelle()
        self.plot_echelle_dP()
        self.axs.echelle.set_xlabel('Period mod {:.{}f} (d)'.format(
            self.echelle_dp, self.decimal_digits_to_display)
        )

        # # Comply with current visibility of PSP in echelle diagram
        if not visible_PSP:
            lst = [
                self.plots.PSP_echelle_scatter_1,
                self.plots.PSP_echelle_scatter_2,
                self.plots.PSP_echelle_scatter_3,
                self.plots.PSP_echelle_dot_1,
                self.plots.PSP_echelle_dot_2,
                self.plots.PSP_echelle_dot_3
            ]
            for artist in lst:
                artist.set_visible(False)
            
    def update_slider(self, ax, slider, vmin, vmax, valinit):
        """Update properties of a matplotlib.widgets.Slider object.
        
        Args:
            ax (matplotlib.axes.Axes):
                Axes object containing the slider.
            slider (matplotlib.widgets.Slider):
                Slider object to update.
            vmin (float):
                Minimum value of the slider.
            vmax (float):
                Maximum value of the slider.
            valinit (float):
                Initial value of the slider.
        """
        slider.valmin = vmin
        slider.valmax = vmax
        ax.set_xlim(slider.valmin, slider.valmax)
        self.update_slider_val(slider, valinit)

    def update_slider_val(self, slider, valinit):
        """Update the value of a matplotlib.widgets.Slider object.
        
        Args:
            slider (matplotlib.widgets.Slider):
                Slider object to update.
            valinit (float):
                Initial value of the slider.
        """
        # This triggers the slider's callback
        slider.set_val(valinit)
        slider.valtext.set_text(slider.valfmt % valinit)

    def sliderAction_amplitude_threshold(self, val):
        """Update the amplitude threshold slider, GUI and the observed periods to 
        fit, according to the value of the slider"""
        # Find periods above threshold and within the window
        pw = self.pw
        pmin = self.window_pmin
        pmax = self.window_pmax
        query = 'period>@pmin and period<@pmax'
        ampl_max = pw.query(query).ampl.max()
        i = (pw.ampl/ampl_max >= val) & (pw.period >= pmin) & (pw.period <= pmax)
        # Leave at least two periods
        if pw[i].period.size >= 2:
            # Apply threshold
            pw.selection = 0
            pw.loc[i, 'selection'] = 1
            # Update p in plot
            xlim = self.axs.pg.get_xlim()
            self.plot_data(echelle_keep_xlim_ylim=True)
            self.axs.pg.set_xlim(xlim)
        # Clear fit information
        self.clear_fit_results()

    def clear_fit_results(self):
        """Clear fit results and GUI elements related to the fit."""
        # Reset fit results
        self.fit = None
        # Clear residual box
        clear_text(self.textBox_residuals)
        # Clear matches
        self.clear_pathCollection(self.plots.matches_scatter)
        self.plots.matches_scatter = None
        # Clear displayed relative errors
        if self.show_relative_error:
            attrs = ['P0', 'dP0', 'Sigma']
            method = 'set_val' if self.writable_text_boxes else 'set_text'
            for attr in attrs:
                val = getattr(self.PSP, attr)
                text = str(round(val, self.decimal_digits_to_display))
                getattr(getattr(self.textBoxes, attr), method)(text)

    def sliderAction_P0(self, val):
        """Update P0, regenerate PSP and replot PSP"""
        self.PSP.P0 = val
        self.PSP.generatePSP()
        self.plot_PSP()
        # Clear previous fit if changes are not triggered from the fit button
        if not self.from_fit_button:
            self.clear_fit_results()
        # Update text boxes
        text = str(round(val, self.decimal_digits_to_display))
        # Display also relative error if requested
        if self.show_relative_error and self.relative_err is not None:
            if self.relative_err.P0 is not None:
                text = text+' ({:.2f}%)'.format(100*self.relative_err.P0)
        attr = 'set_val' if self.writable_text_boxes else 'set_text'
        getattr(self.textBoxes.P0, attr)(text)

    def sliderAction_dP0(self, val):
        """Update dP0, regenerate PSP and replot PSP"""
        self.PSP.dP0 = val
        self.PSP.generatePSP()
        self.plot_PSP()
        # Clear previous fit if changes are not triggered from the fit button
        if not self.from_fit_button:
            self.clear_fit_results()        
        # Update text boxes
        text = str(round(val, self.decimal_digits_to_display))
        # Display also relative error if requested
        if self.show_relative_error and self.relative_err is not None:
            if self.relative_err.dP0 is not None:
                text = text+' ({:.2f}%)'.format(100*self.relative_err.dP0)
        attr = 'set_val' if self.writable_text_boxes else 'set_text'
        getattr(self.textBoxes.dP0, attr)(text)

    def sliderAction_Sigma(self, val):
        """Update Sigma, regenerate PSP and replot PSP"""
        self.PSP.Sigma = val
        self.PSP.generatePSP()
        self.plot_PSP()
        # Clear previous fit if changes are not triggered from the fit button
        if not self.from_fit_button:
            self.clear_fit_results()        
        # Update text boxes
        text = str(round(val, self.decimal_digits_to_display))
        # Display also relative error if requested
        if self.show_relative_error and self.relative_err is not None:
            if self.relative_err.Sigma is not None:
                text = text+' ({:.2f}%)'.format(100*self.relative_err.Sigma)
        attr = 'set_val' if self.writable_text_boxes else 'set_text'
        getattr(self.textBoxes.Sigma, attr)(text)

    def sliderAction_echelle_dP(self, val):
        """Update module period and replot echelle"""
        self.echelle_dp = val
        self.update_axes_echelle(keep_ylim=True)

    def toggle_PSP_visibility(self, event, set=None):
        """Toggle visibility of PSP in the GUI."""
        # Unpack variables
        plots = self.plots
        lst = [
            plots.PSP_echelle_scatter_1,
            plots.PSP_echelle_scatter_2,
            plots.PSP_echelle_scatter_3,
            plots.PSP_echelle_dot_1,
            plots.PSP_echelle_dot_2,
            plots.PSP_echelle_dot_3,
            plots.PSP_dp_dot,
            plots.PSP_pg_vline
        ]
        lst += plots.PSP_dp_lines
        if plots.PSP_pg_lines1 is not None:
            lst += plots.PSP_pg_lines1
        if plots.PSP_pg_lines2 is not None:
            lst += plots.PSP_pg_lines2
        for artist in lst:
            artist.set_visible(not artist.get_visible())

        text = self.buttons.PSP.label.get_text()
        if 'HIDE' in text:
            self.buttons.PSP.label.set_text(text.replace('HIDE','SHOW'))
        elif 'SHOW' in text:
            self.buttons.PSP.label.set_text(text.replace('SHOW','HIDE'))
        else:
            raise ValueError('Unexpected text in button label: '+text)

        self.main_fig.canvas.draw_idle()

    def plot_PSP_in_echelle(self):
        """Overplot periods of the PSP (aka comb) in the echelle plot"""
        # Unpack variables
        plots = self.plots
        ax = self.axs.echelle
        p = self.PSP.p
        echelle_dp = self.echelle_dp
        P0 = self.PSP.P0
        # Clear plot element if it exists
        pathCollections = [
            plots.PSP_echelle_scatter_1,
            plots.PSP_echelle_scatter_2,
            plots.PSP_echelle_scatter_3,
            plots.PSP_echelle_dot_1,
            plots.PSP_echelle_dot_2,
            plots.PSP_echelle_dot_3
        ]
        for pathCollection in pathCollections:
            self.clear_pathCollection(pathCollection)
        # Overplot PSP periods with red circles
        prop = dict(color='red', s=30, zorder=3)
        plots.PSP_echelle_scatter_1 = ax.scatter(p % echelle_dp-echelle_dp, p, **prop)
        plots.PSP_echelle_scatter_2 = ax.scatter(p % echelle_dp+echelle_dp, p, **prop)
        plots.PSP_echelle_scatter_3 = ax.scatter(p % echelle_dp, p, **prop)
        # Over plot P0 with gold circle 
        if self.PSP.nr >= 1:
            prop = dict(color='gold', edgecolor='red', linewidths=0.5, s=30, zorder=3)
            plots.PSP_echelle_dot_1 = ax.scatter(P0 % echelle_dp-echelle_dp, P0, **prop)
            plots.PSP_echelle_dot_2 = ax.scatter(P0 % echelle_dp+echelle_dp, P0, **prop)
            plots.PSP_echelle_dot_3 = ax.scatter(P0 % echelle_dp, P0, **prop)

    def plot_PSP_in_pg(self):
        """Overplot periods of the PSP (aka comb) in the periodogram plot"""

        # Unpack variables
        plots = self.plots
        ax = self.axs.pg
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        p = self.PSP.p
        P0 = self.PSP.P0
        dp = self.PSP.dp
        freq_resolution = self.freq_resolution
        
        self.clear_pathCollection(plots.PSP_pg_vline)
        self.clear_line2D_list(plots.PSP_pg_lines1)
        self.clear_line2D_list(plots.PSP_pg_lines2)

        # Overplot PSP periods with red vertical lines
        prop = dict(color='red', alpha=1, lw=2, zorder=0, ls='-')
        plots.PSP_pg_lines1 = ax.plot(
            np.repeat(p, 3),
            np.tile([0, 1, np.nan], len(p)),
            transform=trans,
            **prop
        )
        # Overplot P0 with a different color, gold
        prop = dict(color='gold', alpha=1, lw=2, zorder=0, ls='-')
        plots.PSP_pg_vline = ax.axvline(
            P0,
            **prop
        )
        
        # Check for unresolved periods
        freq, dfreq = convert_resolution(p, dp)
        unresolved_p = p[dfreq <= freq_resolution]
        if len(unresolved_p) > 0:
            # Overplot PSP unresolved periods with a different color, darkviolet
            prop = dict(color='darkviolet', alpha=1, lw=2, zorder=0, ls='-')
            plots.PSP_pg_lines2 = ax.plot(
                np.repeat(unresolved_p, 3), 
                np.tile([0, 1, np.nan], len(unresolved_p)), 
                transform=trans, 
                **prop
                )
        else:
            # Reinitialize container
            plots.PSP_pg_lines2 = None

    def plot_PSP_in_dp(self):
        """Overplot periods of the PSP (aka comb) in the period spacing plot"""

        # Unpack variables
        plots = self.plots
        ax = self.axs.dp
        p = self.PSP.p
        x = period_for_dP_plot(p, mode=self.pg_mode)
        y = np.diff(p)

        # Clear plot element about PSP if exists
        self.clear_line2D_list(plots.PSP_dp_lines)
        # Ensure that there is at least two periods to the right of P0
        if self.PSP.nr >= 2:
            # Clear plot element about P0 if exists
            self.clear_pathCollection(plots.PSP_dp_dot)

        # Overplot PSP period spacings with red lines
        prop = dict(color='red', alpha=0.8, lw=1, zorder=1, ls='solid')
        plots.PSP_dp_lines = ax.plot(x, y, **prop)
        # Overplot period spacings associated with P0 with a different color, gold
        if self.PSP.nr >= 1:
            i = np.abs(self.PSP.p-self.PSP.P0).argmin()
            period_pair = self.PSP.p[i:i+2]
            x = period_for_dP_plot(period_pair, mode=self.pg_mode)
            y = np.diff(period_pair)
            prop = dict(color='gold', edgecolor='red', linewidths=0.5, s=30, zorder=1)
            plots.PSP_dp_dot = ax.scatter(x, y, **prop)

    def plot_vecinity(self, vals, grids):
        """Generate the 1D and 2D plots of the cost function for all three
        parameters and combinations in the vecinity of the PSP. The vecinity
        is defined by the variables `grid_resolution` and `grid_half_window`.
    
        Args:
            vals (np.ndarray):
                3D array with the cost function values in the vecinity of
                the current PSP.
            grids (dict):
                Dictionary with the grids used to compute the cost function.
                Each key is a 1D array with the grid values for each parameter.
        """

        def plot_1D_vecinity(
            ax,
            grid,
            vals,
            bestVal
        ):
            """Generate a 1D plot of the cost function.
            
            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot in.
                grid (np.ndarray):
                    Grid
                vals (np.ndarray):
                    Values of the grid.
                bestVal (float):
                    Best value of the grid.
            """
            # Clear axes
            ax.clear()
            # Plot 1D values
            prop = dict(ls='solid', lw=1, marker='.', markersize=1, color='black')
            ax.plot(grid, vals, **prop)
            # Plot best-fit
            prop = dict(color='red', ls='solid', lw=1)
            ax.axvline(bestVal, **prop)
                
        def plot_2D_vecinity(
            ax,
            grid1,
            grid2,
            vals,
            bestVal1,
            bestVal2,
            indexing='xy'
        ):
            """Generate a 2D plot of the cost function.
            
            Args:
                ax (matplotlib.axes.Axes):
                    Axes to plot in.
                grid1 (np.ndarray):
                    First 1D grid.
                grid2 (np.ndarray):
                    Second 1D grid.
                vals (np.ndarray):
                    Values for the 2D grid.
                bestVal1 (float):
                    Best value of the first grid.
                bestVal2 (float):
                    Best value of the second grid.
                indexing (str):
                    Indexing of the grid.
            
            Returns:
                cbar (matplotlib.colorbar.Colorbar):
                    Colorbar of the plot.
            """
            # Clear axes
            ax.clear()
            # Plot 2D values
            levels = MaxNLocator(nbins=100).tick_values(vals.min(), vals.max())
            cmap = plt.get_cmap('terrain')
            X, Y = np.meshgrid(grid1, grid2, indexing=indexing)
            cf = ax.contourf(X.T, Y.T, vals, levels=levels, cmap=cmap)
            cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', location='top')
            # Plot bes-fit
            prop = dict(color='red', ls='solid', lw=0.5)
            ax.axvline(bestVal1,  **prop)
            ax.axhline(bestVal2, **prop)
            # Return colorbar
            return cbar

        # Unpack
        axs = self.axs
        PSP = self.PSP
        # Plot P0 vs dP0
        Z = np.log(np.minimum.reduce(vals, axis=2))
        self.P0dP0_cbar = plot_2D_vecinity(
            axs.P0dP0,
            grids['P0'],
            grids['dP0'],
            Z,
            PSP.P0,
            PSP.dP0
        )
        # Plot P0
        plot_1D_vecinity(
            axs.P0,
            grids['P0'],
            np.minimum.reduce(vals, axis=(1, 2)),
            PSP.P0
        )
        # Plot dP0 vs Sigma
        Z = np.log(np.minimum.reduce(vals, axis=0))
        self.dP0Sigma_cbar = plot_2D_vecinity(
            axs.dP0Sigma,
            grids['dP0'],
            grids['Sigma'],
            Z,
            PSP.dP0,
            PSP.Sigma
        )
        # Plot dP0
        plot_1D_vecinity(
            axs.dP0,
            grids['dP0'],
            np.minimum.reduce(vals, axis=(0, 2)),
            PSP.dP0
        )
        # Plot Sigma vs P0
        Z = np.log(np.minimum.reduce(vals, axis=1))
        self.SigmaP0_cbar = plot_2D_vecinity(
            axs.SigmaP0,
            grids['Sigma'],
            grids['P0'],
            Z,
            PSP.Sigma,
            PSP.P0,
            indexing='ij'
        )
        # Plot Sigma
        plot_1D_vecinity(
            axs.Sigma,
            grids['Sigma'],
            np.minimum.reduce(vals, axis=(0, 1)),
            PSP.Sigma
        )
        
    def guess_psp(self): 
        """
        Set attributes by parsing the prewhitenned frequencies and estimating
        P0 and dP0 to generate a first PSP.
        """
        pw = self.pw
        median_dp = np.median(np.diff(pw.period.values))
        dominant_p = pw.query('ampl==ampl.max()').period.values.item()
        self.PSP = LinearPSP(P0=dominant_p, dP0=median_dp)

    def format_main_GUI(self):
        """Format the layout by adding label and tweaks to the axes"""
            
        def format_axes(axs, echelle_dp, title=None):
            """
            Tweak the axes such as labels, ticks, orientation, visibility.

            Args:
                axs (SimpleNamespace):
                    The namespace containing the axes.
                echelle_dp (float):
                    The module of the period spacing.
                title (None, optional):
                    Axes title. Defaults to None.

            Returns:
                None
            """

            def update_echelle_ylim(event_ax, ax=axs.echelle):
                """Update the echelle plot's y-axis accordingly to the x-axis of `ax`."""
                ax.set_ylim(event_ax.get_xlim())

            lst = [
                axs.pg,
                axs.dp,
                axs.echelle,
            ]
            for ax in self.main_fig.axes:
                if ax not in lst:
                    ax.set_navigate(False)

            # Link axis
            axs.dp.sharex(axs.pg)
            axs.p.sharex(axs.pg)
            axs.pg.callbacks.connect('xlim_changed', update_echelle_ylim)
            axs.dp.callbacks.connect('xlim_changed', update_echelle_ylim)
            # Set title
            title = title if title is not None else ''
            axs.p.set_title(f'{title}', y=0.6)
            # Set labels
            axs.pg.set_ylabel('Amplitude (ppt)')
            axs.dp.set_xlabel('Period (d)')
            axs.dp.set_ylabel('$\Delta P$ (d)')
            axs.echelle.set_ylabel('Period (d)')
            axs.echelle.set_xlabel('Period mod {:.{}f} (d)'.format(
                echelle_dp, self.decimal_digits_to_display
            )) # Also defined in `update_axes_echelle` function
            axs.textBoxes.residuals.set_xlabel('Residuals (d)')
            axs.textBoxes.P0.set_xlabel('$P_0$ (d)')
            axs.textBoxes.dP0.set_xlabel('$\Delta P_0$ (d)')
            axs.textBoxes.Sigma.set_xlabel('$\Sigma$')
            # Prune ticks in x axis
            axs.dp.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=6))
            axs.echelle.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            axs.echelle.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            # Prune ticks in y axis
            axs.pg.yaxis.set_major_locator(MaxNLocator(prune=None, nbins=4))
            axs.dp.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=3))
            # Set visibility
            axs.allButtons.axis('off')
            axs.p.axis('off')
            axs.pg.get_xaxis().set_visible(False)
            for ax in vars(axs.textBoxes).values():
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.get_yaxis().set_visible(False)
            # Set ranges
            axs.p.set_ylim(0, 1)

        def format_sliders(axs, sliders, parameter_ranges, PSP):
            """
            Format sliders and its axes. Set the initial values of the sliders,
            hence the initial values of the PSP and (trigger) PSP plots.
            
            Args:
                axs (SimpleNamespace): 
                    namespace containing the axes (matplotlib.axes._subplots.AxesSubplot).
                sliders (SimpleNamespace):
                    namespace containing the sliders (matplotlib.widgets.Slider).
                parameter_ranges (SimpleNamespace):
                    namespace containing the parameter ranges (tuples).
                PSP (LinearPSP):
                    period spacing pattern and its parameters.

            Returns:
                None
            """
            
            def set_slider_values(
                ax,
                slider,
                label,
                vmin,
                vmax,
                valinit,
                valfmt,
                facecolor,
                valstep
            ):
                """
                Set the parameters of `slider` on `ax`.
                
                Args:
                    ax (matplotlib.axes._subplots.AxesSubplot):
                        The axis to apply the slider to.
                    slider (matplotlib.widgets.Slider): 
                        The slider to apply the values to.
                    label (str): 
                        The label of the slider.
                    vmin (float): 
                        The minimum value of the slider.
                    vmax (float): 
                        The maximum value of the slider.
                    valinit (float): 
                        The initial value of the slider.
                    valfmt (str): 
                        The format of the value displayed on the slider.
                    facecolor (str): 
                        The color of the slider.
                    valstep (float): 
                        The step between values on the slider.
                """
                slider.valmin = vmin
                slider.valmax = vmax
                ax.set_xlim(slider.valmin, slider.valmax)
                slider.label.set_text(label)
                slider.set_val(valinit)
                slider.valfmt = valfmt
                slider.valtext.set_text(valfmt % valinit)
                slider.poly.set_fc(facecolor)
                slider.valstep = valstep

            # Make the top and right spines visible
            for ax in axs.__dict__.values():
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
            # Remove vertical lines
            for slider in sliders.__dict__.values():
                l1, l2 = slider.ax.get_lines()
                l1.remove()
            # Amplitude
            vals = {
                'label':'Amplitude threshold (%)', 
                'vmin':0.0, 
                'vmax':1.0, 
                'valfmt':'%1.2f', 
                'valinit':0, 
                'facecolor':'black', 
                'valstep':0.01  # From 1% to 100%
            }
            set_slider_values(axs.ampl, sliders.ampl, **vals)
            # P0
            vals = {
                'label':'$P_0$',
                'vmin':parameter_ranges.P0[0],
                'vmax':parameter_ranges.P0[1],
                'valfmt':'%1.{}f d'.format(self.decimal_digits_to_display),
                'valinit':PSP.P0,
                'facecolor':'red',
                'valstep':self.grid_resolution.P0
            }
            set_slider_values(axs.P0, sliders.P0, **vals)
            # dP0
            vals = {
                'label':'$\Delta P_0$',
                'vmin':parameter_ranges.dP0[0],
                'vmax':parameter_ranges.dP0[1],
                'valfmt':'%1.{}f d'.format(self.decimal_digits_to_display),
                'valinit':PSP.dP0,
                'facecolor':'red',
                'valstep':self.grid_resolution.dP0
            }
            set_slider_values(axs.dP0, sliders.dP0, **vals)
            # Sigma
            vals = {
                'label':'$\Sigma$',
                'vmin':parameter_ranges.Sigma[0],
                'vmax':parameter_ranges.Sigma[1],
                'valfmt':'%1.{}f d'.format(self.decimal_digits_to_display),
                'valinit':0,
                'facecolor':'red',
                'valstep':self.grid_resolution.Sigma
            }
            set_slider_values(axs.Sigma, sliders.Sigma, **vals)
            # Echelle dP
            vals = {
                'label':'Echelle $\Delta P$',
                'vmin':parameter_ranges.dP0[0],
                'vmax':parameter_ranges.dP0[1],
                'valfmt':'%1.{}f d'.format(self.decimal_digits_to_display),
                'valinit':PSP.dP0,
                'facecolor':'dodgerblue',
                'valstep':self.grid_resolution.dP0
            }
            set_slider_values(axs.echelle_dP, sliders.echelle_dP, **vals)

        format_axes(self.axs, self.echelle_dp, self.ID) #TODO?: Better within the if statement?
        format_sliders(self.axs.sliders, self.sliders, self.parameter_ranges, self.PSP)

    def format_secondary_GUI(self):
        """Format the secondary GUI."""
        
        def format_axes(axs):
            """Format the axes of the secondary GUI.

            Args:
                axs (SimpleNamespace):
                    The namespace containing the axes.
            """
            # Link axis
            axs.P0.sharex(axs.P0dP0)
            axs.dP0.sharex(axs.dP0Sigma)
            axs.Sigma.sharex(axs.SigmaP0)

            # Set labels
            axs.P0dP0.set_xlabel('$P_0$')
            axs.P0dP0.set_ylabel('$\Delta P_0$')
            axs.P0.set_xlabel('$P_0$')
            axs.P0.set_ylabel('min ( S | $P_0$ )')
            axs.dP0Sigma.set_xlabel('$\Delta P_0$')
            axs.dP0Sigma.set_ylabel('$\Sigma$')
            axs.dP0.set_xlabel('$\Delta P_0$')
            axs.dP0.set_ylabel('min ( S | $\Delta P_0$ )')
            axs.SigmaP0.set_xlabel('$\Sigma$')
            axs.SigmaP0.set_ylabel('$P_0$')
            axs.Sigma.set_xlabel('$\Sigma$')
            axs.Sigma.set_ylabel('min ( S | $\Sigma$ )')
            
            # Prune ticks in x axis
            axs.P0dP0.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            axs.dP0Sigma.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            axs.SigmaP0.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            axs.P0.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            axs.dP0.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))
            axs.Sigma.xaxis.set_major_locator(MaxNLocator(prune='both', nbins=5))

            # Prune ticks in y axis
            axs.P0dP0.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=4))
            axs.dP0Sigma.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=4))
            axs.SigmaP0.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=4))
            axs.P0.yaxis.set_major_locator( MaxNLocator(prune='upper', nbins=4))
            axs.dP0.yaxis.set_major_locator( MaxNLocator(prune='upper', nbins=4))
            axs.Sigma.yaxis.set_major_locator( MaxNLocator(prune='upper', nbins=4))

            # Set visibility
            lst = [
                axs.P0dP0, 
                axs.dP0Sigma,
                axs.SigmaP0
            ]
            for ax in lst:
                ax.get_xaxis().set_visible(False)
        
        def format_color_bars():
            """Format the color bars of the secondary GUI."""
            # Set labels
            self.P0dP0_cbar.set_label('log min ( S  | $P_0$, $\Delta P_0$ )', labelpad=20)
            self.dP0Sigma_cbar.set_label('log min ( S | $\Delta P_0$, $\Sigma$)', labelpad=20)
            self.SigmaP0_cbar.set_label('log min ( S | $\Sigma$, $P_0$)', labelpad=20)
            # Set ticks
            lst = [self.P0dP0_cbar, self.dP0Sigma_cbar, self.SigmaP0_cbar]
            for cbar in lst:
                cbar.locator = LinearLocator(numticks=4)
                cbar.update_ticks()
        
        format_axes(self.axs)
        format_color_bars()

    def create_secondary_GUI(self):
        """Create figure and axes for the secondary figure and set them as
        attributes of the class."""
        
        if self.secondary_fig is not None:
            plt.close(self.secondary_fig)
        
        # Create figure
        width = 18
        height = 8
        fig = plt.figure(figsize=(width, height))
        gs = fig.add_gridspec(
            nrows=2,
            ncols=3,
            hspace=0.0,
            wspace=0.3,
            height_ratios=[1.2, 1],
            left=0.07,
            right=0.95,
            bottom=0.10,
            top=0.94
        )
        self.axs.P0dP0 = fig.add_subplot(gs[0, 0])
        self.axs.dP0Sigma = fig.add_subplot(gs[0, 1])
        self.axs.SigmaP0 = fig.add_subplot(gs[0, 2])
        self.axs.P0 = fig.add_subplot(gs[1, 0])
        self.axs.dP0 = fig.add_subplot(gs[1, 1])
        self.axs.Sigma = fig.add_subplot(gs[1, 2])
        self.secondary_fig = fig

    def create_main_GUI(self):
        """Creates attributes for figure, axis, sliders, buttons and text boxes
        in the main GUI."""

        def create_figure(width=18, height=8): 
            """
            Create the figure for the GUI and set it as an attribute of the class.
            
            Args:
                width (int, optional): width of the figure in inches. Default is 18.
                height (int, optional): height of the figure in inches. Default is 16.
            """
            figure = plt.figure(figsize=(width, height))
            self.main_fig = figure

        def create_axes(fig): 
            """
            Creates axis objects and set it as an attribute of the class.

            Args:
                fig (matplotlib.figure.Figure): the figure object to create the main axis in.
            """
            
            # Create namespace for axes
            axs = SimpleNamespace()

            # Create axis grid
            gs = fig.add_gridspec(
                nrows=7,
                ncols=2,
                height_ratios=[
                    0.50, # Sliders
                    0.20, # Spacer for potential title
                    0.10, # Period indicator
                    1.00, # Periodogram
                    1.00, # Period spacing
                    0.30, # Spacer before buttons
                    0.25  # Buttons
                ],
                width_ratios=[3.0, 1.2],
                hspace=0.0,
                wspace=0.1,
                left=0.06,
                right=0.98,
                bottom=0.02,
                top=0.98
            )

            # Create axes
            axs.p = fig.add_subplot(gs[2,0])  # Period indicator
            axs.pg = fig.add_subplot(gs[3,0])  # Periodogram
            axs.dp = fig.add_subplot(gs[4,0])  # Period spacing
            axs.echelle = fig.add_subplot(gs[3:5,1])  # Echelle diagram
            axs.allButtons = fig.add_subplot(gs[6,:])  # Buttons

            # Create axes grid for sliders
            gs_sliders = gs[0,:]
            subgs = gs_sliders.subgridspec(
                nrows=5, 
                ncols=3,
                width_ratios=[
                    0.1, # Spacer
                    1.0, # Sliders
                    0.1 # Spacer
                ],
                wspace=0.0,
                hspace=0.3
            )
            # Create axes for sliders
            axs.sliders = SimpleNamespace(
                P0=fig.add_subplot(subgs[0,1]),
                dP0=fig.add_subplot(subgs[1,1]),
                Sigma=fig.add_subplot(subgs[2,1]),
                echelle_dP=fig.add_subplot(subgs[3,1]),
                ampl=fig.add_subplot(subgs[4,1])
            )
            
            # Create axes grid for buttons and text boxes
            gs_buttons = gs[6,:]
            subgs = gs_buttons.subgridspec(
                nrows=2,
                ncols=9,
                wspace=0.1,
                hspace=0.0
            )
            # Create axes for buttons
            axs.buttons = SimpleNamespace(
                fit=fig.add_subplot(subgs[0,0]),
                undo=fig.add_subplot(subgs[1,0]),
                explore=fig.add_subplot(subgs[0,1]),
                save=fig.add_subplot(subgs[1,1]),
                clicking=fig.add_subplot(subgs[0,2]),
                spanning=fig.add_subplot(subgs[1,2]),
                dp_style=fig.add_subplot(subgs[0,3]),
                dp_resolution=fig.add_subplot(subgs[1,3]),
                help=fig.add_subplot(subgs[0,4]),
                PSP=fig.add_subplot(subgs[1,4])
            )
            # Create axes for text boxes
            axs.textBoxes = SimpleNamespace(
                residuals=fig.add_subplot(subgs[0,5]),
                P0=fig.add_subplot(subgs[0,6]),
                dP0=fig.add_subplot(subgs[0,7]),
                Sigma=fig.add_subplot(subgs[0,8])
            )
            
            # Set attribute for axes
            self.axs = axs

        def create_sliders(axsSliders):
            """
            Creates the sliders and add them to the given axes. Set the attribute for the sliders.

            Args:
                axsSliders (SimpleNamespace): the namespace containing the axes for the sliders.
            """

            # Create namespace with sliders            
            sliders = SimpleNamespace(
                ampl=widgets.Slider(axsSliders.ampl, '', 0, 1, handle_style={
                    'facecolor':'white',
                    'edgecolor':'black'
                }),
                echelle_dP=widgets.Slider(axsSliders.echelle_dP, '', 0, 1, handle_style={
                    'facecolor':'white',
                    'edgecolor':'black'
                }),
                P0=widgets.Slider(axsSliders.P0, '', 0, 1, handle_style={
                    'facecolor':'gold',
                    'edgecolor':'black'
                }),
                dP0=widgets.Slider(axsSliders.dP0, '', 0, 1, handle_style={
                    'facecolor':'gold',
                    'edgecolor':'black'
                }),
                Sigma=widgets.Slider(axsSliders.Sigma, '', 0, 1, handle_style={
                    'facecolor':'white',
                    'edgecolor':'black'
                })
            )
            # Add callbacks to sliders
            sliders.ampl.on_changed(self.sliderAction_amplitude_threshold)
            sliders.echelle_dP.on_changed(self.sliderAction_echelle_dP)
            sliders.P0.on_changed(self.sliderAction_P0)
            sliders.dP0.on_changed(self.sliderAction_dP0)
            sliders.Sigma.on_changed(self.sliderAction_Sigma)
            # Set attribute
            self.sliders = sliders

        def create_buttons(axsButtons):
            """Creates buttons and add them to the given axes. Set the attribute for the buttons.

            Args:
                axsButtons (SimpleNamespace): the namespace containing the axes for the buttons.
            """
            # Namespace for the buttons
            buttons = SimpleNamespace(
                fit=widgets.Button(axsButtons.fit, 'Fit'),
                explore=widgets.Button(axsButtons.explore, 'Explore'),
                save=widgets.Button(axsButtons.save, 'Save'),
                help=widgets.Button(axsButtons.help, 'Help'),
                undo=widgets.Button(axsButtons.undo, 'Undo'),
                clicking=widgets.Button(axsButtons.clicking, 'Clicker mode OFF'),
                spanning=widgets.Button(axsButtons.spanning, 'Spanner mode OFF'),
                dp_style=widgets.Button(axsButtons.dp_style, 'HIDE $\Delta$P line'),
                dp_resolution=widgets.Button(axsButtons.dp_resolution, 'HIDE $\Delta$P resolution'),
                PSP=widgets.Button(axsButtons.PSP, 'HIDE PSP')
            )
            # Add callbacks to buttons
            buttons.fit.on_clicked(self.fitPSP)
            buttons.explore.on_clicked(self.explore_PSP_vecinity)
            buttons.save.on_clicked(self.save)
            buttons.help.on_clicked(self.help)
            buttons.undo.on_clicked(self.undo)
            buttons.clicking.on_clicked(self.toggle_clicking_mode)
            buttons.spanning.on_clicked(self.toggle_spanning_mode)
            buttons.dp_style.on_clicked(self.toggle_dp_line_style)
            buttons.dp_resolution.on_clicked(self.toggle_dp_resolution_line)
            buttons.PSP.on_clicked(self.toggle_PSP_visibility)
            # Set attribute
            self.buttons = buttons

        def create_textboxes(axsTextBoxes):
            """Creates text boxes and add them to the given axes. Set the attribute for the text boxes.

            - Text box Residuals: Display the residuals of the fit. Read-only.
            - Text box P0: Display value (and error if requested). Read and write or read-only.
            - Text box dP0: Display value (and error if requested). Read and write or read-only.
            - Text box Sigma: Display value (and error if requested). Read and write or read-only.
            
            Note: The use of writable text boxes slows down the GUI. Therefore,
            they are disabled by default (see variable `writable_text_boxes`). More
            information on StackOverflow: https://stackoverflow.com/a/59613370/9290590

            Args:
                axsTextBoxes (SimpleNamespace): the namespace containing the axes for the text boxes.
            """

            # Namespace for the textboxes
            textBoxes = SimpleNamespace()

            yText = 0
            sizeText = 11

            # Textbox for residuals
            text_alignment = (0.9, 0.5)
            props = dict(ha='right', va='center', transform=axsTextBoxes.residuals.transAxes)
            self.textBox_residuals = axsTextBoxes.residuals.text(*text_alignment, '', **props)

            # Textboxes            
            attrs = ['P0', 'dP0', 'Sigma']
            if self.writable_text_boxes:
                props = dict(color='lightgoldenrodyellow', textalignment='right')
                for attr in attrs:
                    setattr(textBoxes, attr, widgets.TextBox(getattr(axsTextBoxes,attr), "", initial='0', **props))
                    self.connections_main_GUI.append(getattr(textBoxes,attr).on_submit(getattr(self,'read_box_'+attr)))
            else:
                for attr in attrs:
                    props = dict(ha='right', va='center', transform=getattr(axsTextBoxes,attr).transAxes)
                    setattr(textBoxes, attr, getattr(axsTextBoxes,attr).text(*text_alignment, '', **props))

            self.textBoxes = textBoxes

        create_figure()
        create_axes(self.main_fig)
        create_sliders(self.axs.sliders)
        create_buttons(self.axs.buttons)
        create_textboxes(self.axs.textBoxes)

    def plot_p_indicator(self):
        """Plot black/grey triangles on top of the periodogram to indicate which
        periods are used for the fit."""
        # Clear plot if it exists
        self.clear_pathCollection(self.plots.p_scatter_0)
        # Unpack
        pw = self.pw
        ax = self.axs.p
        x = pw.period.values
        y = np.repeat(0.2, x.size)
        color = self.color_period_on_off[pw.selection.values]
        prop = dict(c=color, marker=7, alpha=1, zorder=2, picker=5)
        self.plots.p_scatter_0 = ax.scatter(x, y, **prop)

    def plot_pg(self):
        """Plot the periodogram of the light curve in question."""
        ax = self.axs.pg
        # Plot the periodogram of the light curve
        x = self.pg.period
        y = self.pg.ampl
        prop = dict(lw=1, color='black', zorder=3)
        ax.plot(x, y, **prop)
        # Mark level zero
        prop = dict(lw=0.5, color='gray', ls='-')
        ax.axhline(0, **prop)

    def plot_p_as_lines_in_pg(self):
        """Plot the observed periods to be fitted as vertical dotted lines in the periodogram."""
        # Clear plot if it exists
        self.clear_line2D_list(self.plots.p_lines)
        # Unpack
        pw = self.pw
        p = pw.query('selection==1').period.values
        ax = self.axs.pg
        trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
        prop = dict(lw=1, ls='dotted', color='black')
        line = ax.plot(np.repeat(p, 3), np.tile([0, 1, np.nan], len(p)), transform=trans, **prop)
        self.plots.p_lines = line

    def plot_dp(self):
        """Plot the period spacing of the observed periods to be fitted."""
        # Clear plot if it exists
        self.clear_line2D_list(self.plots.dp_lines)
        # Unpack
        pw = self.pw
        p = pw.query('selection==1').period.values
        ax = self.axs.dp
        x = period_for_dP_plot(p, mode=self.pg_mode)
        y = np.diff(p)
        prop = dict(lw=1, color='black', ls=self.linestyle_dp, marker='.', zorder=2)
        self.plots.dp_lines = ax.plot(x, y, **prop)
        prop = dict(lw=0.5, color='gray', ls='-')
        ax.axhline(0, **prop)

    def plot_echelle(
        self,
        keep_xlim=False,
        keep_ylim=False,
        keep_xlim_ylim=False
    ):
        """Plot the echelle diagram of the observations.
        
        Args:
            keep_xlim (bool,optinal):
                Do not update the x-axis limits of the plot. Defaults to False.
            keep_ylim (bool,optinal):
                Do not update the y-axis limits of the plot. Defaults to False.
            keep_xlim_ylim (bool,optinal):
                Do not update the x- and y-axis limits of the plot. Defaults to False.
        """
        # Unpack
        pw = self.pw
        ax = self.axs.echelle
        p = pw.period.values
        echelle_dp = self.echelle_dp
        ampl = pw.ampl.values
        selection = pw.selection.values
        # Clear plot element if it exists
        markers = [
            self.plots.echelle_vline1,
            self.plots.echelle_vline2,
            self.plots.p_scatter_1,
            self.plots.p_scatter_2,
            self.plots.p_scatter_3
        ]
        for marker in markers:
            self.clear_pathCollection(marker)
        # Plotted range
        if keep_xlim_ylim:
            keep_xlim = True
            keep_ylim = True
        if keep_xlim:
            xlim = ax.get_xlim()
        if keep_ylim:
            ylim = ax.get_ylim()
        # Plot echelle
        prop = dict(s=100.*(ampl/ampl.max()), color=self.color_period_on_off[selection], zorder=2, picker=5)
        self.plots.p_scatter_1 = ax.scatter(p % echelle_dp-echelle_dp, p, **prop)
        self.plots.p_scatter_2 = ax.scatter(p % echelle_dp+echelle_dp, p, **prop)
        self.plots.p_scatter_3 = ax.scatter(p % echelle_dp, p, **prop)
        # Set axis limits
        ax.set_xlim(xlim) if keep_xlim else ax.set_xlim(-echelle_dp, 2*echelle_dp)
        ax.set_ylim(ylim) if keep_ylim else ax.set_ylim(p.min(), p.max())
        # Separe the 3 plotted echelles
        prop = dict(ls='solid', color='dodgerblue', lw=1, zorder=0)
        self.plots.echelle_vline1 = ax.axvline(0, **prop)
        self.plots.echelle_vline2 = ax.axvline(echelle_dp, **prop)

    def plot_echelle_dP(self):
        """Plot the module period on the period spacing plot as a horizontal line"""
        self.clear_pathCollection(self.plots.dp_hline)
        ax = self.axs.dp
        prop = dict(lw=1, color='dodgerblue', ls='-', zorder=0)
        self.plots.dp_hline = ax.axhline(self.echelle_dp, **prop, label='Echelle $\Delta P$')
        self.axs.dp.legend()

    def plot_resolution_in_dp(self, df):
        """Plot the frequency resolution in the period spacing plot.
        
        Args:
            df (float):
                Frequency resolution.
        """
        # Dummy periods
        pmin = self.pw.period.min()
        pmax = self.pw.period.max()
        p = np.linspace(pmin, pmax, 1000)
        # Compute the period resolution
        dp = get_period_resolution(p, df)
        # Plot
        prop = dict(color='darkviolet', alpha=1, lw=1, zorder=0, ls='-')
        self.plots.dp_resolution = self.axs.dp.plot(p, dp, **prop, label='Limit resolution')
        self.axs.dp.legend()
        
if __name__ == '__main__':
    from example import flossyExample 
    app = QApplication(sys.argv)
    flossyExample()
    sys.exit(app.exec_())
