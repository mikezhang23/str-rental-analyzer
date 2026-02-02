"""
Market Explorer Page
====================

Interactive map showing Airbnb listings colored by performance metrics.
Users can filter by location, bedrooms, price, and explore the market visually.
"""

import streamlit as st
import pandas as pd
import folium                              # Map library
from streamlit_folium import folium_static # Renders folium maps in Streamlit
from utils.data_loader import load_listings, get_market_stats


# ---------------------------------------------------------------------
# PAGE CONFIG
# Note: set_page_config can only be called once per app, and we already
# call it in app.py. For pages, we skip it or Streamlit handles it.
# ---------------------------------------------------------------------

st.title("🗺️ Market Explorer")
st.markdown("Explore Las Vegas Airbnb listings by location and performance.")


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

listings = load_listings()
stats = get_market_stats(listings)


# ---------------------------------------------------------------------
# SIDEBAR FILTERS
# These let users narrow down what they see on the map.
# ---------------------------------------------------------------------

st.sidebar.header("Filters")

# Bedroom filter
# get unique values, sort them, convert to list for the multiselect
bedroom_options = sorted(listings['bedrooms'].dropna().unique().astype(int).tolist())

selected_bedrooms = st.sidebar.multiselect(
    "Bedrooms",                            # Label
    options=bedroom_options,               # Available choices
    default=bedroom_options                # Start with all selected
)

# Price range filter
# We use a slider with min/max from the data
min_price = int(listings['price_clean'].min())
max_price = int(listings['price_clean'].max())

price_range = st.sidebar.slider(
    "Nightly Rate ($)",                    # Label
    min_value=min_price,                   # Slider minimum
    max_value=max_price,                   # Slider maximum
    value=(min_price, max_price)           # Default selection (tuple = range)
)

# Revenue filter (minimum annual revenue)
min_revenue = st.sidebar.number_input(
    "Minimum Annual Revenue ($)",
    min_value=0,
    max_value=int(listings['annual_revenue'].max()),
    value=0,
    step=5000
)

# Neighborhood filter (if you have this column)
if 'neighbourhood_cleansed' in listings.columns:
    neighborhoods = sorted(listings['neighbourhood_cleansed'].dropna().unique().tolist())
    selected_neighborhoods = st.sidebar.multiselect(
        "Neighborhoods",
        options=neighborhoods,
        default=neighborhoods
    )
else:
    selected_neighborhoods = None


# ---------------------------------------------------------------------
# APPLY FILTERS
# Create a filtered dataframe based on user selections.
# ---------------------------------------------------------------------

# Start with full dataset
filtered = listings.copy()

# Filter by bedrooms
filtered = filtered[filtered['bedrooms'].isin(selected_bedrooms)]

# Filter by price range
filtered = filtered[
    (filtered['price_clean'] >= price_range[0]) & 
    (filtered['price_clean'] <= price_range[1])
]

# Filter by minimum revenue
filtered = filtered[filtered['annual_revenue'] >= min_revenue]

# Filter by neighborhood (if applicable)
if selected_neighborhoods is not None:
    filtered = filtered[filtered['neighbourhood_cleansed'].isin(selected_neighborhoods)]

# Show how many listings match
st.markdown(f"**{len(filtered):,}** listings match your filters")

# Stop here if no matches
if len(filtered) == 0:
    st.warning("⚠️ No properties match your current filters. Try:")
    st.markdown("""
    - Expanding the price range
    - Selecting more bedroom options
    - Lowering the minimum revenue
    - Including more neighborhoods
    """)
    st.stop()  # This prevents the rest of the page from running

# ---------------------------------------------------------------------
# METRICS ROW
# Show summary stats for the FILTERED data
# ---------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Median ADR", f"${filtered['price_clean'].median():.0f}")
    
with col2:
    st.metric("Median Occupancy", f"{filtered['occupancy_rate'].median()*100:.0f}%")
    
with col3:
    st.metric("Median Revenue", f"${filtered['annual_revenue'].median():,.0f}")
    
with col4:
    st.metric("90th % Revenue", f"${filtered['annual_revenue'].quantile(0.90):,.0f}")


# ---------------------------------------------------------------------
# MAP VISUALIZATION
# Create an interactive map with listings as colored markers.
# ---------------------------------------------------------------------

st.subheader("Listing Map")

# Check if we have lat/long data
if 'latitude' not in filtered.columns or 'longitude' not in filtered.columns:
    st.error("Missing latitude/longitude data in listings.")

# Check if filtered data is empty
elif len(filtered) == 0:
    st.warning("No properties match your filters. Try adjusting the criteria.")

else:
    # Limit points for performance (too many markers = slow map)
    MAX_POINTS = 1000
    
    if len(filtered) > MAX_POINTS:
        st.info(f"Showing {MAX_POINTS:,} of {len(filtered):,} listings for performance.")
        map_data = filtered.sample(n=MAX_POINTS, random_state=42)
    else:
        map_data = filtered
    
    # Remove rows with missing coordinates
    map_data = map_data.dropna(subset=['latitude', 'longitude'])
    
    # Create base map centered on Las Vegas
    # We calculate the center from the data
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],  # Center point
        zoom_start=11,                       # Initial zoom level
        tiles='CartoDB positron'             # Clean, light map style
    )
    
    # ---------------------------------------------------------------------
    # COLOR MARKERS BY REVENUE
    # We'll create a simple color scale: red (low) -> yellow -> green (high)
    # ---------------------------------------------------------------------
    
    def get_color(revenue, median, top_90):
        """
        Assign color based on revenue performance.
        
        Args:
            revenue: This listing's annual revenue
            median: Market median revenue
            top_90: 90th percentile revenue
            
        Returns:
            str: Color name for the marker
        """
        if revenue >= top_90:
            return 'darkgreen'      # Top performer
        elif revenue >= median:
            return 'green'          # Above average
        elif revenue >= median * 0.5:
            return 'orange'         # Below average
        else:
            return 'red'            # Poor performer
    
    # Calculate thresholds from filtered data
    median_rev = filtered['annual_revenue'].median()
    top_90_rev = filtered['annual_revenue'].quantile(0.90)
    
    # Add markers to map
    for idx, row in map_data.iterrows():
        
        # Get color for this listing
        color = get_color(row['annual_revenue'], median_rev, top_90_rev)
        
        # Create popup text (what shows when you click)
        popup_text = f"""
        <b>${row['price_clean']:.0f}/night</b><br>
        {int(row['bedrooms'])} bed<br>
        Occupancy: {row['occupancy_rate']*100:.0f}%<br>
        Revenue: ${row['annual_revenue']:,.0f}/yr
        """
        
        # Add circle marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,                          # Size of circle
            color=color,                       # Border color
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey;">
        <p><strong>Revenue Performance</strong></p>
        <p><span style="color: darkgreen;">●</span> Top 10%</p>
        <p><span style="color: green;">●</span> Above Median</p>
        <p><span style="color: orange;">●</span> Below Median</p>
        <p><span style="color: red;">●</span> Bottom Tier</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Render the map in Streamlit
    folium_static(m, width=1000, height=600)


# ---------------------------------------------------------------------
# DATA TABLE (Optional - expandable)
# Let users see the raw data if they want
# ---------------------------------------------------------------------

with st.expander("View Raw Data"):
    # Select columns to display
    display_cols = ['neighbourhood_cleansed', 'bedrooms', 'price_clean', 
                    'occupancy_rate', 'annual_revenue']
    
    # Only include columns that exist
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    # Format and display
    st.dataframe(
        filtered[display_cols].sort_values('annual_revenue', ascending=False),
        use_container_width=True
    )