# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:12:48 2024

@author: avajdi
"""

import numpy as np

def cord_to_CRofland(longitude, latitude):
    """
    Convert longitude and latitude to grid column and row indices.

    Parameters:
        longitude (float or array-like): Longitude value(s) in the range 0-360 or -180 to 180.
        latitude (float or array-like): Latitude value(s) in the range -90 to 90.

    Returns:
        tuple: A tuple containing:
            - col (int or array-like): Column index.
            - row (int or array-like): Row index.
    """
    # Convert latitude to column index
    col = np.ceil(721 - 4 * (latitude + 0.125 + 90)).astype(int)
    
    # Convert longitude to row index
    row = np.maximum(np.ceil((longitude + 0.125) % 360 / 0.25), 1).astype(int)
    
    return col, row

# Example usage
"""
longitude = np.arange(-180, 180, 0.05)  # Example range of longitudes
latitude = np.arange(90, -90.05, -0.05)  # Example latitude
col, row = cord_to_CRofland(longitude, latitude)

print(f"Column indices: {col}")
print(f"Row indices: {row}")
"""