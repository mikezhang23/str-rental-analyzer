"""
Short-Term Rental Investment Analyzer
=====================================

Main entry point for the Streamlit application.
This file creates the home page and sets up the app configuration.

To run locally:
    streamlit run app.py
"""

import streamlit as st                 # The web framework

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# This MUST be the first Streamlit command in your script.
# It sets the browser tab title, icon, and layout.
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="STR Investment Analyzer",    # Browser tab title
    page_icon="🏠",                          # Browser tab icon (emoji or image path)
    layout="wide",                           # Use full width of browser
    initial_sidebar_state="expanded"         # Sidebar starts open
)


# ---------------------------------------------------------------------
# CUSTOM CSS (Optional)
# Streamlit allows injecting custom CSS to style the app.
# This is advanced - skip if you want to keep it simple.
# ---------------------------------------------------------------------

st.markdown("""
    <style>
    /* Make the main content area have some padding */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Style metric cards */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)           # unsafe_allow_html=True allows HTML/CSS


# ---------------------------------------------------------------------
# MAIN PAGE CONTENT
# This is what users see when they first open the app.
# ---------------------------------------------------------------------

# Title with emoji
st.title("🏠 Short-Term Rental Investment Analyzer")

# Subtitle/description
st.markdown("""
**Las Vegas Market Analysis with Causal Inference**

This tool helps real estate investors identify profitable Airbnb 
investment opportunities using data-driven analysis and statistical methods.
""")

# Add some space
st.divider()                           # Horizontal line


# ---------------------------------------------------------------------
# KEY METRICS ROW
# Display headline statistics using Streamlit's metric component.
# We import from our utils module.
# ---------------------------------------------------------------------

from utils.data_loader import load_listings, get_market_stats

# Load data (cached, so fast after first load)
listings = load_listings()
stats = get_market_stats(listings)

# Create 4 columns for metrics
col1, col2, col3, col4 = st.columns(4)

# Display metrics in each column
with col1:
    st.metric(
        label="Total Listings Analyzed",   # Small text above
        value=f"{stats['total_listings']:,}"  # Big number (formatted with commas)
    )

with col2:
    st.metric(
        label="Median Nightly Rate",
        value=f"${stats['median_adr']:.0f}"   # Format as dollars
    )

with col3:
    st.metric(
        label="Median Annual Revenue",
        value=f"${stats['median_revenue']:,.0f}"
    )

with col4:
    st.metric(
        label="90th Percentile Revenue",
        value=f"${stats['revenue_90th']:,.0f}"
    )


# ---------------------------------------------------------------------
# NAVIGATION INSTRUCTIONS
# Guide users to the sidebar pages.
# ---------------------------------------------------------------------

st.divider()

st.markdown("""
### 📍 How to Use This Tool

Use the **sidebar** on the left to navigate between pages:

1. **🗺️ Market Explorer** - Interactive map to explore listings by location and ROI
2. **📊 Property Analyzer** - Compare specific properties against market benchmarks  
3. **🧪 Amenity Impact** - Causal analysis of how amenities affect revenue
4. **💰 ROI Calculator** - Estimate returns for potential investments
5. **📖 Methodology** - Statistical methods and data sources explained

---

### 🎯 Key Insights from Our Analysis

- **2-bedroom properties** deliver the highest ROI (up to 18%)
- **Pools** increase revenue by 93% (causal estimate)
- **Unincorporated Areas (The Strip)** outperforms other neighborhoods
- **Top 10% operators** earn 3x the median revenue
""")


# ---------------------------------------------------------------------
# SIDEBAR CONTENT
# The sidebar appears on all pages. Good for global filters or info.
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    Built by Michael Zhang ([mikezhang23.github.io])
    
    Data Sources:
    - Inside Airbnb
    - MLS Listings
    - Booking.com API
    
    [GitHub Repo](https://github.com/mikezhang23/str-rental-analyzer)
    """)