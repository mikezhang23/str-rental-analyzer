"""
Data loading utilities for the STR Investment Analyzer.
This module centralizes all data loading and caching logic.
Streamlit reruns the entire script on every interaction, so caching is critical for performance.
"""

import pandas as pd                    # Data manipulation library
import streamlit as st                 # Streamlit for caching decorator
from pathlib import Path               # Cross-platform file paths
from utils.stats import prepare_amenity_flags


# ---------------------------------------------------------------------
# CACHING EXPLANATION:
# @st.cache_data tells Streamlit to store the result of this function.
# On subsequent runs, if the inputs haven't changed, Streamlit returns
# the cached result instead of re-executing the function.
# This makes the app FAST - we don't reload CSVs on every click.
# ---------------------------------------------------------------------

@st.cache_data                         # Cache the result of this function
def load_listings():
    """
    Load the cleaned Airbnb listings data.
    
    Returns:
        pd.DataFrame: Listings with columns like price_clean, 
                      occupancy_rate, annual_revenue, amenities, etc.
    """
    # Path(__file__) gets the path to THIS file (data_loader.py)
    # .parent gets the directory containing this file (utils/)
    # .parent again gets the project root (str-rental-analyzer/)
    # Then we navigate to data/listings_clean.csv
    
    data_path = Path(__file__).parent.parent / "data" / "listings_clean.csv"
    
    # Read the CSV into a pandas DataFrame
    df = pd.read_csv(data_path)
    
    # Ensure numeric columns are correct type
    # (CSVs sometimes load numbers as strings)
    numeric_cols = ['price_clean', 'occupancy_rate', 'annual_revenue', 
                    'bedrooms', 'latitude', 'longitude']
    
    for col in numeric_cols:
        if col in df.columns:          # Check column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to number
    
    return df


@st.cache_data
def load_mls():
    """
    Load MLS (home sales) data.
    
    Returns:
        pd.DataFrame: Home listings with price, bedrooms, location, etc.
    """
    data_path = Path(__file__).parent.parent / "data" / "mls_data.csv"
    df = pd.read_csv(data_path)
    
    return df


@st.cache_data
def get_market_stats(df):
    """
    Calculate market-level statistics for benchmarking.
    
    This creates summary statistics that users can compare
    individual properties against.
    
    Args:
        df: Listings DataFrame
        
    Returns:
        dict: Market statistics including medians, percentiles, etc.
    """
    stats = {
        'median_adr': df['price_clean'].median(),           # Middle ADR
        'median_occupancy': df['occupancy_rate'].median(),  # Middle occupancy
        'median_revenue': df['annual_revenue'].median(),    # Middle revenue
        
        # Percentiles for "good" and "excellent" benchmarks
        'revenue_75th': df['annual_revenue'].quantile(0.75),
        'revenue_90th': df['annual_revenue'].quantile(0.90),
        
        # Counts for context
        'total_listings': len(df),
    }
    
    return stats

@st.cache_data
def load_mls_active():
    """
    Load and combine all active MLS listings.
    
    These are homes currently for sale that we can analyze
    as potential STR investments.
    
    Returns:
        pd.DataFrame: Combined active listings with cleaned columns
    """
    # List of MLS files to combine
    mls_files = [
        "data/mls-active-2bd-500kmax.csv",
        "data/mls-active-3bd-500kmax.csv",
        "data/mls-active-4bd-500kmax.csv",
        "data/mls-active-5bd-667kmax.csv",
        "data/mls-active-6bd-868kmax.csv",
    ]
    
    # Load and combine all files
    dfs = []
    for file in mls_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except FileNotFoundError:
            continue  # Skip missing files silently
    
    # Return empty dataframe if no files found
    if len(dfs) == 0:
        return pd.DataFrame()
    
    # Combine into single dataframe
    combined = pd.concat(dfs, ignore_index=True)
    
    # Clean price column (remove $ and commas)
    combined['price_clean'] = (
        combined['Current Price']
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .astype(float)
    )
    
    # Clean square footage
    combined['sqft'] = (
        combined['Approx Liv Area']
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    
    # Rename columns for consistency
    combined = combined.rename(columns={
        'Beds Total': 'bedrooms',
        'Baths Total': 'bathrooms',
        'Zip Code': 'zip_code',
        'Pv Pool': 'has_pool',
        'Spa': 'has_spa',
        'Year Built': 'year_built',
        'DOM': 'days_on_market',
        'Address': 'address'
    })
    
    # Convert pool/spa to boolean
    combined['has_pool'] = combined['has_pool'] == 'Y'
    combined['has_spa'] = combined['has_spa'] == 'Y'
    
    return combined

@st.cache_data
def load_listings_with_amenities():
    """Load listings with parsed amenity flags."""
    df = load_listings()
    df = prepare_amenity_flags(df)
    return df