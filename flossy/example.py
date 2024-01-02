import sys
from types import SimpleNamespace
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from flossy import flossyGUI
from PyQt5.QtWidgets import QApplication

def handle_input_data(TIC):
    """Custom function to handle the input data in the example folder
    (url). #TODO: fill in the url.
    
    It reads the prewhitened and periodogram CSV files and extracts the
    relevant columns into numpy arrays, then it gives astropy units to each
    array. Finally, it also generates the title to show in the GUI.
    
    Returns a SimpleNamespace object with the following attributes:
        - pw_periods: array of periods from the prewhitened data
        - pw_e_periods: array of errors on the periods from the prewhitened data
        - pw_amplitudes: array of amplitudes from the prewhitened data
        - pg_periods: array of periods from the periodogram data
        - pg_amplitudes: array of amplitudes from the periodogram data
        - title: title to show in the GUI
    """
    
    # Prewhitened CSV file
    pw_file = f'flossy/example_data/pw/pw_tic{TIC}.csv'
    # Periodogram CSV file
    pg_file = f'flossy/example_data/pg/pg_tic{TIC}.csv'
    # Read the CSV files
    pw = pd.read_csv(pw_file) 
    pg = pd.read_csv(pg_file)
    # Extract relevant columns from the CSV files
    pw_periods = 1/pw.frequency.values
    pw_e_periods = pw.e_frequency.values * pw_periods**2
    pw_amplitudes = pw.amp.values
    pg_periods = 1/pg.freq.values
    pg_amplitudes = pg.amp.values
    # Give astropy units to the just extracted columns 
    pw_periods *= u.day
    pw_e_periods *= u.day
    pw_amplitudes = pw_amplitudes * 1e-3 * u.dimensionless_unscaled # ppt
    pg_periods *= u.day
    pg_amplitudes = pg_amplitudes * 1e-3 * u.dimensionless_unscaled # ppt
    # Give a title to the figure
    title = f'TIC {TIC}'
    data = SimpleNamespace(
        pw_periods=pw_periods,
        pw_e_periods=pw_e_periods,
        pw_amplitudes=pw_amplitudes,
        pg_periods=pg_periods,
        pg_amplitudes=pg_amplitudes,
        title=title
    )
    return data

def flossyExample():
    data = handle_input_data(TIC=374944608)
    # Create the interface
    GUI = flossyGUI(
        pw_periods=data.pw_periods,
        pw_e_periods=data.pw_e_periods,
        pw_amplitudes=data.pw_amplitudes,
        pg_periods=data.pg_periods,
        pg_amplitudes=data.pg_amplitudes,
        ID=data.title,
        freq_resolution=1/u.yr
    )
    # Run the interface using a context manager
    with GUI as flossy:
        app = QApplication([])
        plt.show()
        sys.exit(app.exec_())
        
if __name__ == '__main__':
    flossyExample()