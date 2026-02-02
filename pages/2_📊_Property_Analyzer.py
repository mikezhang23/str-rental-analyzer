"""
Property Analyzer Page
======================

Analyze active MLS listings as potential STR investments.
Filter properties, select from table, and see projected performance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from utils.data_loader import load_listings_with_amenities, load_mls_active
from utils.stats import get_all_amenity_impacts, get_recommendations_for_property


st.title("📊 Property Analyzer")
st.markdown("Analyze active listings as potential short-term rental investments.")


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

# Airbnb data (for market benchmarks)
airbnb = load_listings_with_amenities()

# Active MLS listings (homes for sale)
mls_active = load_mls_active()

# Check if MLS data loaded
if len(mls_active) == 0:
    st.error("No MLS data found. Please add MLS CSV files to the data/ folder.")
    st.stop()

st.markdown(f"**{len(mls_active):,}** active listings available for analysis")


# ---------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------

st.sidebar.header("Filter Properties")

# Bedroom filter
bedroom_options = sorted(mls_active['bedrooms'].dropna().unique().astype(int).tolist())
selected_bedrooms = st.sidebar.multiselect(
    "Bedrooms",
    options=bedroom_options,
    default=bedroom_options
)

# Price range filter
min_price = int(mls_active['price_clean'].min())
max_price = int(mls_active['price_clean'].max())

price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price),
    step=10000,
    format="$%d"
)

# Zip code filter
zip_codes = sorted(mls_active['zip_code'].dropna().unique().astype(str).tolist())
selected_zips = st.sidebar.multiselect(
    "Zip Codes",
    options=zip_codes,
    default=[]  # Empty = all zips
)

# Pool filter
pool_filter = st.sidebar.selectbox(
    "Pool",
    options=["Any", "Yes", "No"]
)

# Square footage filter
min_sqft = int(mls_active['sqft'].min())
max_sqft = int(mls_active['sqft'].max())

sqft_range = st.sidebar.slider(
    "Square Footage",
    min_value=min_sqft,
    max_value=max_sqft,
    value=(min_sqft, max_sqft),
    step=100
)


# ---------------------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------------------

filtered = mls_active.copy()

# Bedroom filter
filtered = filtered[filtered['bedrooms'].isin(selected_bedrooms)]

# Price filter
filtered = filtered[
    (filtered['price_clean'] >= price_range[0]) &
    (filtered['price_clean'] <= price_range[1])
]

# Zip code filter (only apply if user selected specific zips)
if len(selected_zips) > 0:
    filtered = filtered[filtered['zip_code'].astype(str).isin(selected_zips)]

# Pool filter
if pool_filter == "Yes":
    filtered = filtered[filtered['has_pool'] == True]
elif pool_filter == "No":
    filtered = filtered[filtered['has_pool'] == False]

# Square footage filter
filtered = filtered[
    (filtered['sqft'] >= sqft_range[0]) &
    (filtered['sqft'] <= sqft_range[1])
]

# Show count
st.markdown(f"**{len(filtered):,}** properties match your filters")

# Stop if no matches
if len(filtered) == 0:
    st.warning("No properties match your filters. Try adjusting the criteria.")
    st.stop()


# ---------------------------------------------------------------------
# PROPERTY TABLE (Selectable)
# ---------------------------------------------------------------------

st.subheader("Select a Property")

# Prepare display dataframe
display_df = filtered[[
    'address', 'zip_code', 'bedrooms', 'bathrooms', 
    'sqft', 'price_clean', 'has_pool', 'year_built', 'days_on_market'
]].copy()

# Format columns for display
display_df = display_df.rename(columns={
    'address': 'Address',
    'zip_code': 'Zip',
    'bedrooms': 'Beds',
    'bathrooms': 'Baths',
    'sqft': 'SqFt',
    'price_clean': 'Price',
    'has_pool': 'Pool',
    'year_built': 'Year',
    'days_on_market': 'DOM'
})

# Sort by price
display_df = display_df.sort_values('Price')

# Create selection using data_editor (allows row selection)
# We add a selection column
display_df.insert(0, 'Select', False)

edited_df = st.data_editor(
    display_df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select a property to analyze",
            default=False,
        ),
        "Price": st.column_config.NumberColumn(
            "Price",
            format="$%d"
        ),
        "SqFt": st.column_config.NumberColumn(
            "SqFt",
            format="%d"
        ),
        "Pool": st.column_config.CheckboxColumn(
            "Pool"
        )
    },
    disabled=['Address', 'Zip', 'Beds', 'Baths', 'SqFt', 'Price', 'Pool', 'Year', 'DOM']
)

# Get selected row(s)
selected_rows = edited_df[edited_df['Select'] == True]

if len(selected_rows) == 0:
    st.info("👆 Select a property from the table above to see its analysis.")
    st.stop()

if len(selected_rows) > 1:
    st.warning("Please select only one property at a time.")
    st.stop()


# ---------------------------------------------------------------------
# GET SELECTED PROPERTY DATA
# ---------------------------------------------------------------------

selected_address = selected_rows['Address'].iloc[0]
selected_property = filtered[filtered['address'] == selected_address].iloc[0]


# ---------------------------------------------------------------------
# PROPERTY DETAILS
# ---------------------------------------------------------------------

st.divider()
st.subheader(f"Analysis: {selected_property['address']}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Price", f"${selected_property['price_clean']:,.0f}")

with col2:
    st.metric("Bedrooms", f"{int(selected_property['bedrooms'])}")

with col3:
    st.metric("Square Feet", f"{selected_property['sqft']:,.0f}")

with col4:
    pool_status = "Yes" if selected_property['has_pool'] else "No"
    st.metric("Pool", pool_status)


# ---------------------------------------------------------------------
# PROJECTED REVENUE (Based on Airbnb comps)
# ---------------------------------------------------------------------

st.subheader("Projected Revenue")

# Get Airbnb comps (same bedroom count)
bedroom_count = int(selected_property['bedrooms'])
airbnb_comps = airbnb[airbnb['bedrooms'] == bedroom_count]

if len(airbnb_comps) == 0:
    st.warning(f"No Airbnb data available for {bedroom_count} bedroom properties.")
else:
    # Calculate revenue projections
    median_revenue = airbnb_comps['annual_revenue'].median()
    revenue_75th = airbnb_comps['annual_revenue'].quantile(0.75)
    revenue_90th = airbnb_comps['annual_revenue'].quantile(0.90)
    
    # If property has pool, adjust projections upward (93% lift from your analysis)
    pool_multiplier = 1.93 if selected_property['has_pool'] else 1.0
    
    # Adjusted projections
    adj_median = median_revenue * pool_multiplier
    adj_75th = revenue_75th * pool_multiplier
    adj_90th = revenue_90th * pool_multiplier
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Operator",
            f"${adj_median:,.0f}/yr",
            help="Median revenue for this bedroom count"
        )
    
    with col2:
        st.metric(
            "Good Operator (75th %)",
            f"${adj_75th:,.0f}/yr",
            help="Top 25% of operators achieve this"
        )
    
    with col3:
        st.metric(
            "Excellent Operator (90th %)",
            f"${adj_90th:,.0f}/yr",
            help="Top 10% of operators achieve this"
        )
    
    if selected_property['has_pool']:
        st.caption("📊 Projections include 93% revenue lift for having a pool (based on our amenity analysis).")


# ---------------------------------------------------------------------
# MARKET RANKING (Percentile Gauges)
# ---------------------------------------------------------------------

st.subheader("Market Ranking")
st.caption("How this property compares to the MLS market.")

# Calculate percentiles vs other MLS listings
price_percentile = stats.percentileofscore(
    mls_active['price_clean'].dropna(),
    selected_property['price_clean']
)

sqft_percentile = stats.percentileofscore(
    mls_active['sqft'].dropna(),
    selected_property['sqft']
)

# Price per sqft
mls_active['price_per_sqft'] = mls_active['price_clean'] / mls_active['sqft']
selected_ppsf = selected_property['price_clean'] / selected_property['sqft']

ppsf_percentile = stats.percentileofscore(
    mls_active['price_per_sqft'].dropna(),
    selected_ppsf
)


def create_gauge(value, title):
    """
    Create a clean gauge chart showing percentile.
    
    Args:
        value: Percentile (0-100)
        title: Chart title
        
    Returns:
        plotly figure object
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            # No background colors (steps removed)
        }
    ))
    
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=25, r=25))
    
    return fig


col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(
        create_gauge(price_percentile, "Price"),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        create_gauge(sqft_percentile, "Square Footage"),
        use_container_width=True
    )

with col3:
    st.plotly_chart(
        create_gauge(ppsf_percentile, "Price/SqFt"),
        use_container_width=True
    )


# ---------------------------------------------------------------------
# ROI PROJECTION
# ---------------------------------------------------------------------

st.subheader("ROI Projection")

# Mortgage assumption (6% annual cost of home price)
mortgage_rate = 0.06
annual_mortgage = selected_property['price_clean'] * mortgage_rate

# Calculate ROI for each operator level
if len(airbnb_comps) > 0:
    roi_median = ((adj_median - annual_mortgage) / selected_property['price_clean']) * 100
    roi_75th = ((adj_75th - annual_mortgage) / selected_property['price_clean']) * 100
    roi_90th = ((adj_90th - annual_mortgage) / selected_property['price_clean']) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "normal" if roi_median > 0 else "inverse"
        st.metric(
            "ROI (Average)",
            f"{roi_median:.1f}%",
            delta=f"${adj_median - annual_mortgage:,.0f} net"
        )
    
    with col2:
        st.metric(
            "ROI (Good Operator)",
            f"{roi_75th:.1f}%",
            delta=f"${adj_75th - annual_mortgage:,.0f} net"
        )
    
    with col3:
        st.metric(
            "ROI (Excellent)",
            f"{roi_90th:.1f}%",
            delta=f"${adj_90th - annual_mortgage:,.0f} net"
        )
    
    st.caption(f"Based on 6% annual cost (mortgage + taxes + insurance) = ${annual_mortgage:,.0f}/year")
    
# ---------------------------------------------------------------------
# AMENITY RECOMMENDATIONS (Based on Causal Analysis)
# ---------------------------------------------------------------------

st.subheader("📈 Amenity Recommendations")

# Get causal impacts for all amenities
amenity_impacts = get_all_amenity_impacts(airbnb)

if len(amenity_impacts) == 0:
    st.warning("Unable to calculate amenity impacts. Insufficient data.")
else:
    # Determine which amenities this property has
    current_amenities = []
    if selected_property['has_pool']:
        current_amenities.append('Pool')
    if selected_property.get('has_spa', False):
        current_amenities.append('Hot Tub')
    
    # Get recommendations (only positive, significant impacts)
    recommendations = get_recommendations_for_property(
        selected_property, 
        amenity_impacts,
        current_amenities
    )
    
    if len(recommendations) == 0:
        st.info("No high-confidence amenity recommendations for this property. This could mean the property already has key amenities, or our analysis didn't find statistically significant positive impacts.")
    else:
        st.markdown("**Top opportunities to increase revenue:**")
        
        for i, rec in enumerate(recommendations[:5], 1):
            confidence_emoji = "🟢" if rec['confidence'] == 'High' else "🟡"
            
            st.markdown(f"**{i}. {rec['amenity']}** {confidence_emoji}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Revenue Lift",
                    f"${rec['revenue_lift']:,.0f}/yr",
                    delta=f"{rec['pct_lift']:+.0f}%"
                )
            
            with col2:
                st.metric(
                    "Install Cost",
                    f"${rec['install_cost']:,.0f}"
                )
            
            with col3:
                payback_display = f"{rec['payback_years']:.1f} years" if rec['payback_years'] < 100 else "N/A"
                st.metric(
                    "Payback",
                    payback_display
                )
            
            st.divider()
        
        st.caption("🟢 High confidence | 🟡 Medium confidence")
    
    # Detailed view with sortable numeric columns
    with st.expander("View All Amenity Impacts"):
        # Create display dataframe - keep numeric for sorting
        display_impacts = amenity_impacts.copy()
        
        # Single "Recommendation" column that combines impact direction + confidence
        def get_recommendation(row):
            if row['revenue_impact'] > 0 and row['p_value'] < 0.01:
                return "✅ Strongly Recommended"
            elif row['revenue_impact'] > 0 and row['p_value'] < 0.05:
                return "✅ Recommended"
            elif row['revenue_impact'] > 0:
                return "⚠️ Possibly Helpful (needs more data)"
            elif row['p_value'] < 0.05:
                return "❌ Not Recommended"
            else:
                return "❓ Inconclusive"
        
        display_impacts['Recommendation'] = display_impacts.apply(get_recommendation, axis=1)
        
        # Round numeric values for display
        display_impacts['revenue_impact'] = display_impacts['revenue_impact'].round(0).astype(int)
        display_impacts['pct_lift'] = display_impacts['pct_lift'].round(1)
        display_impacts['n_matched'] = display_impacts['n_matched'].astype(int)
        
        # Select columns (removed separate Confidence column)
        display_df = display_impacts[[
            'amenity', 'revenue_impact', 'pct_lift', 'n_matched', 'Recommendation'
        ]].copy()
        
        display_df.columns = ['Amenity', 'Revenue Impact ($)', 'Lift (%)', 'Sample Size', 'Recommendation']
        
        # Sort by revenue impact descending
        display_df = display_df.sort_values('Revenue Impact ($)', ascending=False)
        
        # Use column_config for proper formatting while keeping sortability
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Revenue Impact ($)": st.column_config.NumberColumn(
                    "Revenue Impact ($)",
                    format="$%d"
                ),
                "Lift (%)": st.column_config.NumberColumn(
                    "Lift (%)",
                    format="%.1f%%"
                ),
                "Sample Size": st.column_config.NumberColumn(
                    "Sample Size",
                    format="%d"
                )
            }
        )
        
        st.markdown("---")
        st.markdown("**Recommendation Guide:**")
        
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.success("**✅ Strongly Recommended** — Positive impact, very high confidence")
            st.success("**✅ Recommended** — Positive impact, high confidence")
            st.warning("**⚠️ Possibly Helpful** — Positive trend, but needs more data")
        with rec_col2:
            st.error("**❌ Not Recommended** — No positive impact detected")
            st.info("**❓ Inconclusive** — Not enough evidence either way")
    
    # Methodology explanation - proper web layout
    with st.expander("📊 How We Calculate These Recommendations"):
        
        st.markdown("### The Problem with Simple Comparisons")
        
        st.warning("""
        **Why we can't just compare "pool vs no pool" directly:**
        
        Properties with pools tend to be in better neighborhoods, larger, and have more bedrooms. 
        So a simple comparison would attribute the neighborhood benefit to the pool — that's misleading.
        """)
        
        st.markdown("---")
        st.markdown("### Our Solution: Propensity Score Matching")
        
        st.markdown("PSM creates fair comparisons by finding similar properties.")
        
        # Step 1
        st.markdown("#### Step 1: Calculate Propensity Scores")
        st.markdown("""
        For each property, we calculate the probability it would have a pool based on its **physical characteristics**:
        - Number of bedrooms
        - Number of bathrooms  
        - Guest capacity
        - Neighborhood
        """)
        
        st.info("""
        **Why we exclude nightly price:** Price is part of the outcome (revenue = price × occupancy). 
        A pool lets you charge higher prices — that's part of its value we want to measure, not control away.
        """)
        
        # Step 2
        st.markdown("#### Step 2: Match Similar Properties")
        st.markdown("""
        We pair each pool property with a non-pool property that has a similar propensity score.
        
        **What does this mean?** If two properties have the same score (say, 0.65), they're similar in all the ways 
        that typically predict pool ownership — similar size, neighborhood, and capacity. But one has a pool and one doesn't.
        """)
        
        # Step 3
        st.markdown("#### Step 3: Compare Revenue")
        st.markdown("We compare annual revenue between matched pairs. The average difference is our estimate of the pool's true impact.")
        
        st.markdown("---")
        st.markdown("### Visual Example")
        
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.error("**Without PSM (Biased)**")
            st.markdown("""
            **Pool Properties:**  
            Mostly large, near Strip, expensive  
            Average revenue: **$50,000/year**
            
            **Non-Pool Properties:**  
            Mixed sizes, mixed areas  
            Average revenue: **$25,000/year**
            
            ❌ Naive conclusion: Pool adds $25,000!  
            *Reality: Most is location/size, not pool*
            """)
        
        with col_after:
            st.success("**With PSM (Fair)**")
            st.markdown("""
            **Pool Property:**  
            4-bed, Strip area, sleeps 8  
            Revenue: **$48,000/year**
            
            **Matched Non-Pool:**  
            4-bed, Strip area, sleeps 8  
            Revenue: **$41,000/year**
            
            ✅ Fair conclusion: Pool adds ~$7,000
            """)
        
        st.markdown("---")
        st.markdown("### How We Determine Confidence")
        
        conf_col1, conf_col2 = st.columns(2)
        
        with conf_col1:
            st.markdown("""
            | Level | Certainty |
            |-------|-----------|
            | **Very High** | 99%+ certain impact is real |
            | **High** | 95%+ certain impact is real |
            """)
        
        with conf_col2:
            st.markdown("""
            | Level | Certainty |
            |-------|-----------|
            | **Moderate** | 90%+ certain, could be noise |
            | **Low** | Not enough evidence |
            """)
        
        st.markdown("---")
        st.markdown("### Why Some Results Are Inconclusive")
        
        st.markdown("""
        An amenity might show inconclusive results if:
        
        1. **Small sample size** — Not enough properties with that amenity
        2. **Uneven distribution** — All properties with that amenity are in one area  
        3. **No good matches** — Can't find similar properties without the amenity
        """)
        
        st.markdown("---")
        st.markdown("### Important Caveats")
        
        st.warning("""
        - These are **estimates** based on market data, not guarantees
        - Your results depend on **execution** (photography, pricing, guest experience)
        - Market conditions **change over time**
        - Installation costs are **rough estimates** — get actual quotes
        """)

# ---------------------------------------------------------------------
# REVENUE DISTRIBUTION
# ---------------------------------------------------------------------

st.subheader("Revenue Distribution")
st.caption(f"How {bedroom_count}-bedroom Airbnb properties perform (note: right-skewed distribution is normal - most earn modest returns, top operators significantly outperform)")

if len(airbnb_comps) > 0:
    fig = px.histogram(
        airbnb_comps,
        x='annual_revenue',
        nbins=30,
        labels={'annual_revenue': 'Annual Revenue ($)'}
    )
    
    # Add vertical lines for percentiles
    fig.add_vline(
        x=median_revenue,
        line_dash="solid",
        line_color="gray",
        annotation_text="Median",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=revenue_75th,
        line_dash="dash",
        line_color="blue",
        annotation_text="75th %",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=revenue_90th,
        line_dash="dash",
        line_color="green",
        annotation_text="90th %",
        annotation_position="top"
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Annual Revenue ($)",
        yaxis_title="Number of Properties"
    )
    
    st.plotly_chart(fig, use_container_width=True)