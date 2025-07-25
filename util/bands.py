"""
File: bands.py
Author: Chuncheng Zhang
Date: 2025-07-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Define bands for frequency analysis.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-07-22 ------------------------
# Requirements and constants


# %% ---- 2025-07-22 ------------------------
# Function and class
class Bands:
    """
    Define frequency bands for analysis.
    """

    def __init__(self):
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            # 'beta': (12, 30),
            'beta': (15, 24),  # The project uses this beta band.
            'gamma': (30, 45),
            'all': (0.1, 30)
        }

    def get_band(self, band_name: str):
        """
        Get the frequency range for a given band name.
        """
        return self.bands.get(band_name, None)

    def mk_band_range(self, band_name: str):
        """
        Make a range of frequencies for a given band name.
        """
        l_freq, h_freq = self.bands[band_name]

        return range(l_freq, h_freq + 1)


# %% ---- 2025-07-22 ------------------------
# Play ground


# %% ---- 2025-07-22 ------------------------
# Pending


# %% ---- 2025-07-22 ------------------------
# Pending
