"""
Statistical Analysis Module
===========================

Contains functions for causal inference using Propensity Score Matching.
These functions calculate the true impact of amenities on revenue,
controlling for confounding variables.

Why Propensity Score Matching?
------------------------------
Simple comparison (pool vs no pool) is biased because:
- Properties with pools tend to be in better neighborhoods
- Properties with pools tend to be larger
- Properties with pools tend to have other amenities too

PSM creates "matched pairs" of similar properties where the only
meaningful difference is the amenity we're studying. This isolates
the causal effect of the amenity itself.
"""

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

# For Propensity Score Matching
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


@st.cache_data
def prepare_amenity_flags(df):
    """
    Parse amenities column and create boolean flags for key amenities.
    Also adds control variables for PSM.
    """
    import ast
    import math
    
    # Make a copy
    df = df.copy()
    
    # Parse amenities string into list (if stored as string)
    if 'amenities' in df.columns and df['amenities'].dtype == 'object':
        def safe_parse(x):
            try:
                if pd.isna(x):
                    return []
                if isinstance(x, list):
                    return x
                return ast.literal_eval(x)
            except:
                return []
        
        df['amenities_list'] = df['amenities'].apply(safe_parse)
    elif 'amenities_list' not in df.columns:
        df['amenities_list'] = [[]] * len(df)
    
    # Define amenities to analyze
    amenities_to_check = {
        'pool': ['pool', 'swimming'],
        'hot_tub': ['hot tub', 'jacuzzi', 'spa'],
        'sauna': ['sauna'],
        'game_room': ['game room', 'game console', 'arcade', 'pool table', 'ping pong'],
        'theater': ['theater', 'theatre', 'projector', 'home cinema'],
        'gym': ['gym', 'fitness', 'exercise'],
        'outdoor_kitchen': ['outdoor kitchen', 'bbq', 'grill'],
        'fire_pit': ['fire pit', 'firepit'],
        'ev_charger': ['ev charger', 'electric vehicle'],
    }
    
    def has_amenity(amenities_list, keywords):
        if not amenities_list:
            return False
        amenities_lower = [a.lower() for a in amenities_list]
        return any(
            keyword in amenity 
            for amenity in amenities_lower 
            for keyword in keywords
        )
    
    # Create boolean columns for each amenity
    for amenity_name, keywords in amenities_to_check.items():
        df[f'has_{amenity_name}'] = df['amenities_list'].apply(
            lambda x: has_amenity(x, keywords)
        )
    
    # =========================================
    # ADD CONTROL VARIABLES FOR PSM
    # =========================================
    
    # 1. Distance from Strip (center of Las Vegas Strip)
    STRIP_LAT = 36.1147
    STRIP_LON = -115.1728
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        def calc_distance(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                return None
            # Simple Euclidean distance (good enough for this scale)
            # ~69 miles per degree latitude, ~54.6 miles per degree longitude at this latitude
            lat_diff = (row['latitude'] - STRIP_LAT) * 69
            lon_diff = (row['longitude'] - STRIP_LON) * 54.6
            return math.sqrt(lat_diff**2 + lon_diff**2)
        
        df['distance_from_strip'] = df.apply(calc_distance, axis=1)
    
    # 2. Property type dummies (top categories)
    if 'property_type' in df.columns:
        df['is_entire_home'] = df['property_type'].str.contains('home|house', case=False, na=False).astype(int)
        df['is_condo'] = df['property_type'].str.contains('condo', case=False, na=False).astype(int)
    
    # 3. Room type dummies
    if 'room_type' in df.columns:
        df['is_entire_place'] = (df['room_type'] == 'Entire home/apt').astype(int)
        df['is_private_room'] = (df['room_type'] == 'Private room').astype(int)
    
    # 4. Superhost flag
    if 'host_is_superhost' in df.columns:
        df['is_superhost'] = (df['host_is_superhost'] == 't').astype(int)
    
    # 5. Professional host (has multiple listings)
    if 'calculated_host_listings_count' in df.columns:
        df['is_professional_host'] = (df['calculated_host_listings_count'] >= 3).astype(int)
    
    # 6. Minimum nights (cap at 30 for outliers)
    if 'minimum_nights' in df.columns:
        df['min_nights_capped'] = df['minimum_nights'].clip(upper=30)
    
    # 7. Neighborhood dummies (top 7)
    if 'neighbourhood_cleansed' in df.columns:
        top_hoods = df['neighbourhood_cleansed'].value_counts().head(7).index.tolist()
        for i, hood in enumerate(top_hoods):
            df[f'neighborhood_{i}'] = (df['neighbourhood_cleansed'] == hood).astype(int)
    
    return df


@st.cache_data
def calculate_propensity_scores(df, treatment_col, covariate_cols):
    """
    Calculate propensity scores for a treatment variable.
    
    Propensity score = probability of receiving treatment given covariates.
    
    Args:
        df: DataFrame with treatment and covariates
        treatment_col: Name of binary treatment column (e.g., 'has_pool')
        covariate_cols: List of columns to use as covariates
        
    Returns:
        Array of propensity scores (one per row)
    """
    # Prepare data
    df_clean = df.dropna(subset=[treatment_col] + covariate_cols)
    
    X = df_clean[covariate_cols].copy()
    y = df_clean[treatment_col].astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    # Get propensity scores (probability of treatment=1)
    propensity_scores = model.predict_proba(X_scaled)[:, 1]
    
    return propensity_scores, df_clean.index


def match_propensity_scores(df, treatment_col, propensity_scores, caliper=0.05):
    """
    Match treated units to control units based on propensity scores.
    
    Uses nearest neighbor matching with caliper (maximum distance).
    
    Args:
        df: DataFrame
        treatment_col: Name of treatment column
        propensity_scores: Array of propensity scores
        caliper: Maximum allowed difference in propensity scores
        
    Returns:
        DataFrame with matched pairs
    """
    # Add propensity scores to dataframe
    df = df.copy()
    df['propensity_score'] = propensity_scores
    
    # Separate treated and control groups
    treated = df[df[treatment_col] == True].copy()
    control = df[df[treatment_col] == False].copy()
    
    if len(treated) == 0 or len(control) == 0:
        return pd.DataFrame()  # Can't match if one group is empty
    
    # Use nearest neighbors to find matches
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[['propensity_score']])
    
    # Find nearest control for each treated unit
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    # Create matched pairs (only those within caliper)
    matched_treated = []
    matched_control = []
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist[0] <= caliper:  # Within caliper
            matched_treated.append(treated.iloc[i])
            matched_control.append(control.iloc[idx[0]])
    
    if len(matched_treated) == 0:
        return pd.DataFrame()
    
    # Combine into matched dataset
    matched_treated_df = pd.DataFrame(matched_treated)
    matched_treated_df['match_group'] = 'treated'
    
    matched_control_df = pd.DataFrame(matched_control)
    matched_control_df['match_group'] = 'control'
    
    matched = pd.concat([matched_treated_df, matched_control_df], ignore_index=True)
    
    return matched


@st.cache_data
def calculate_amenity_ate(_df, amenity_col, outcome_col='annual_revenue', 
                          covariate_cols=None):
    """..."""
    
    import streamlit as st
    
    df = _df.copy()
    
    # DEBUG
    st.write(f"DEBUG: Starting with {len(df)} rows")
    st.write(f"DEBUG: distance_from_strip nulls: {df['distance_from_strip'].isna().sum() if 'distance_from_strip' in df.columns else 'NOT FOUND'}")
    st.write(f"DEBUG: Columns available: {[c for c in df.columns if c.startswith('is_') or c.startswith('neighborhood_') or c == 'distance_from_strip']}")

    # Default covariates - comprehensive list
    if covariate_cols is None:
        covariate_cols = []
        
        # Core property characteristics
        core_cols = ['bedrooms', 'accommodates', 'bathrooms']
        covariate_cols.extend([c for c in core_cols if c in df.columns])
        
        # Distance from Strip (most important!)
        if 'distance_from_strip' in df.columns:
            covariate_cols.append('distance_from_strip')
        
        # Property type
        type_cols = ['is_entire_home', 'is_condo', 'is_entire_place', 'is_private_room']
        covariate_cols.extend([c for c in type_cols if c in df.columns])
        
        # Host characteristics
        host_cols = ['is_superhost', 'is_professional_host']
        covariate_cols.extend([c for c in host_cols if c in df.columns])
        
        # Minimum nights
        if 'min_nights_capped' in df.columns:
            covariate_cols.append('min_nights_capped')
        
        # Neighborhood dummies (only the numeric ones we created)
        neighborhood_cols = [c for c in df.columns if c.startswith('neighborhood_') and c[-1].isdigit()]
        covariate_cols.extend(neighborhood_cols)
        
       # Only keep columns that exist and have no issues
        covariate_cols = [c for c in covariate_cols if c in df.columns]
        
        # DEBUG: Show what we're using
        st.write(f"DEBUG: Covariates selected: {covariate_cols}")
        
        # Check for NaN in each covariate
        for col in covariate_cols:
            null_count = df[col].isna().sum()
            st.write(f"DEBUG: {col} - nulls: {null_count}, dtype: {df[col].dtype}")

    """
    Calculate Average Treatment Effect (ATE) of an amenity using PSM.
    
    Args:
        _df: Listings DataFrame (underscore prefix tells Streamlit not to hash)
        amenity_col: Name of amenity column (e.g., 'has_pool')
        outcome_col: Name of outcome variable (default: 'annual_revenue')
        covariate_cols: Columns to control for. If None, uses defaults.
        
    Returns:
        dict with ATE estimate, confidence interval, and diagnostics
    """
    # Make a copy to avoid modifying original
    df = _df.copy()
    
    # Default covariates - ONLY pre-treatment variables
    # We do NOT include price or reviews because:
    # - Price is part of the outcome (revenue = price × occupancy)
    # - A pool lets you charge MORE, which is part of its value
    # - Reviews are affected by having the amenity (more bookings = more reviews)
    
    if covariate_cols is None:
        # Primary covariates (property characteristics that exist before amenity)
        covariate_cols = ['bedrooms', 'accommodates']
        
        # Add square footage if available
        if 'sqft' in df.columns:
            covariate_cols.append('sqft')
        
        # Add bathrooms if available
        if 'bathrooms' in df.columns:
            covariate_cols.append('bathrooms')
        
        # Add location encoding if available (critical for proper matching)
        if 'neighbourhood_cleansed' in df.columns:
            # Create dummy variables for neighborhoods
            df = pd.get_dummies(df, columns=['neighbourhood_cleansed'], prefix='hood', drop_first=True)
            hood_cols = [c for c in df.columns if c.startswith('hood_')]
            covariate_cols.extend(hood_cols)
        
        # Only use columns that exist
        covariate_cols = [c for c in covariate_cols if c in df.columns]
        
        # Add other relevant covariates if available
        optional_covariates = ['bathrooms', 'review_scores_rating', 'number_of_reviews']
        for col in optional_covariates:
            if col in df.columns:
                covariate_cols.append(col)
        
        # Only use columns that exist and have no issues
        covariate_cols = [c for c in covariate_cols if c in df.columns]

    # Need at least one covariate
    if len(covariate_cols) == 0:
        return {'ate': None, 'error': 'No valid covariates available'}
    
    # FORCE numeric conversion for all relevant columns
    try:
        for col in covariate_cols + [outcome_col]:
            if col in df.columns:
                # Convert to string first, then to numeric (handles mixed types)
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Also ensure amenity column is boolean
        df[amenity_col] = df[amenity_col].astype(bool)
        
    except Exception as e:
        return {'ate': None, 'error': f'Data conversion error: {str(e)}'}
    
    # Prepare data - drop rows with missing values
    required_cols = [amenity_col, outcome_col] + covariate_cols
    df_clean = df.dropna(subset=required_cols).copy()
    
    # DEBUG
    st.write(f"DEBUG: After dropna: {len(df_clean)} rows (dropped {len(df) - len(df_clean)})")
    
    # Need sufficient sample size
    n_treated = df_clean[amenity_col].sum()
    n_control = len(df_clean) - n_treated
    
    if n_treated < 30 or n_control < 30:
        return {
            'ate': None,
            'error': f'Insufficient sample size (treated: {n_treated}, control: {n_control})'
        }
    
    try:
        # Calculate propensity scores
        X = df_clean[covariate_cols].values.astype(float)  # Force float
        y = df_clean[amenity_col].astype(int).values       # Force int
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        # Get propensity scores
        propensity_scores = model.predict_proba(X_scaled)[:, 1]
        
        # Add propensity scores to dataframe
        df_clean = df_clean.copy()
        df_clean['propensity_score'] = propensity_scores
        
        # Separate treated and control groups
        treated = df_clean[df_clean[amenity_col] == True].copy()
        control = df_clean[df_clean[amenity_col] == False].copy()
        
        if len(treated) == 0 or len(control) == 0:
            return {'ate': None, 'error': 'Empty treatment or control group'}
        
        # Use nearest neighbors to find matches
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(control[['propensity_score']].values)
        
        distances, indices = nn.kneighbors(treated[['propensity_score']].values)
        
        # Create matched pairs (within caliper of 0.05)
        caliper = 0.05
        matched_treated_outcomes = []
        matched_control_outcomes = []
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist[0] <= caliper:
                matched_treated_outcomes.append(float(treated.iloc[i][outcome_col]))
                matched_control_outcomes.append(float(control.iloc[idx[0]][outcome_col]))
        
        if len(matched_treated_outcomes) < 10:
            return {'ate': None, 'error': f'Only {len(matched_treated_outcomes)} matches found'}
        
        # Convert to arrays
        treated_outcomes = np.array(matched_treated_outcomes)
        control_outcomes = np.array(matched_control_outcomes)
        
        # Calculate ATE
        ate = float(treated_outcomes.mean() - control_outcomes.mean())
        
        # Calculate confidence interval using t-test
        t_stat, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
        
        # Standard error
        se = np.sqrt(
            (treated_outcomes.std()**2 / len(treated_outcomes)) + 
            (control_outcomes.std()**2 / len(control_outcomes))
        )
        
        # 95% confidence interval
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        # Percentage lift
        baseline = float(control_outcomes.mean())
        pct_lift = (ate / baseline) * 100 if baseline > 0 else 0
        
        return {
            'ate': ate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'pct_lift': pct_lift,
            'p_value': float(p_value),
            'n_matched_pairs': len(treated_outcomes),
            'n_treated_total': int(n_treated),
            'n_control_total': int(n_control),
            'treated_mean': float(treated_outcomes.mean()),
            'control_mean': float(control_outcomes.mean()),
            'baseline_revenue': baseline
        }
        
    except Exception as e:
        return {'ate': None, 'error': f'Calculation error: {str(e)}'}

@st.cache_data
def get_all_amenity_impacts(_df):
    """
    Calculate causal impact for all amenities.
    """
    import streamlit as st
    
    # List of amenity columns to analyze
    amenity_cols = [col for col in _df.columns if col.startswith('has_')]
    
    # Remove has_availability if present (not a real amenity)
    amenity_cols = [col for col in amenity_cols if col != 'has_availability']
    
    
    results = []
    
    for amenity_col in amenity_cols:
        amenity_name = amenity_col.replace('has_', '').replace('_', ' ').title()
        
        # Calculate ATE
        impact = calculate_amenity_ate(_df, amenity_col)
    
        if impact.get('ate') is not None:
            results.append({
                'amenity': amenity_name,
                'amenity_col': amenity_col,
                'revenue_impact': impact['ate'],
                'pct_lift': impact['pct_lift'],
                'ci_lower': impact['ci_lower'],
                'ci_upper': impact['ci_upper'],
                'p_value': impact['p_value'],
                'significant': impact['p_value'] < 0.05,
                'n_matched': impact['n_matched_pairs'],
                'baseline_revenue': impact['baseline_revenue']
            })
    
    # Convert to DataFrame
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('revenue_impact', ascending=False)
    else:
        results_df = pd.DataFrame(columns=[
            'amenity', 'amenity_col', 'revenue_impact', 'pct_lift',
            'ci_lower', 'ci_upper', 'p_value', 'significant', 
            'n_matched', 'baseline_revenue'
        ])
    
    return results_df


def get_recommendations_for_property(property_row, amenity_impacts, current_amenities=None):
    """
    Generate specific recommendations for a property based on causal analysis.
    
    Only recommends amenities with POSITIVE and STATISTICALLY SIGNIFICANT impacts.
    
    Args:
        property_row: Series with property details
        amenity_impacts: DataFrame from get_all_amenity_impacts()
        current_amenities: List of amenities the property already has
        
    Returns:
        List of recommendation dicts
    """
    if current_amenities is None:
        current_amenities = []
    
    recommendations = []
    
    # Installation cost estimates (rough)
    install_costs = {
        'Pool': 45000,
        'Hot Tub': 10000,
        'Sauna': 6000,
        'Game Room': 5000,
        'Theater': 8000,
        'Gym': 3000,
        'Outdoor Kitchen': 15000,
        'Fire Pit': 2000,
        'Ev Charger': 1500,
    }
    
    for _, row in amenity_impacts.iterrows():
        amenity_name = row['amenity']
        
        # Skip if property already has this amenity
        if amenity_name.lower() in [a.lower() for a in current_amenities]:
            continue
        
        # ONLY recommend if POSITIVE impact AND statistically significant
        if row['revenue_impact'] <= 0:
            continue
            
        if not row['significant']:
            continue
        
        # Calculate ROI of adding this amenity
        install_cost = install_costs.get(amenity_name, 5000)
        annual_revenue_gain = row['revenue_impact']
        years_to_payback = install_cost / annual_revenue_gain if annual_revenue_gain > 0 else float('inf')
        five_year_roi = ((annual_revenue_gain * 5) - install_cost) / install_cost * 100
        
        recommendations.append({
            'amenity': amenity_name,
            'revenue_lift': annual_revenue_gain,
            'pct_lift': row['pct_lift'],
            'install_cost': install_cost,
            'payback_years': years_to_payback,
            'five_year_roi': five_year_roi,
            'confidence': 'High' if row['p_value'] < 0.01 else 'Medium'
        })
    
    # Sort by 5-year ROI
    recommendations = sorted(recommendations, key=lambda x: x['five_year_roi'], reverse=True)
    
    return recommendations