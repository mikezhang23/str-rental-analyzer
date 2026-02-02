"""
Amenity Impact Page
===================

Deep dive into causal analysis of amenity impacts on revenue.
Explore how each amenity affects performance using Propensity Score Matching.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import load_listings_with_amenities
from utils.stats import get_all_amenity_impacts, calculate_amenity_ate


st.title("🧪 Amenity Impact Analysis")
st.markdown("Understand which amenities actually drive revenue using causal inference.")


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

airbnb = load_listings_with_amenities()
amenity_impacts = get_all_amenity_impacts(airbnb)


# ---------------------------------------------------------------------
# OVERVIEW SECTION
# ---------------------------------------------------------------------

st.subheader("Overview")

st.markdown("""
Not all amenities are created equal. Some genuinely increase revenue, while others 
are just correlated with better-performing properties. We use **Propensity Score Matching** 
to isolate the true causal impact of each amenity.
""")

# Key stats
col1, col2, col3, col4 = st.columns(4)

# Count recommended amenities
recommended = amenity_impacts[
    (amenity_impacts['revenue_impact'] > 0) & 
    (amenity_impacts['significant'] == True)
]

inconclusive = amenity_impacts[
    (amenity_impacts['significant'] == False)
]

not_recommended = amenity_impacts[
    (amenity_impacts['revenue_impact'] <= 0) & 
    (amenity_impacts['significant'] == True)
]

with col1:
    st.metric("Amenities Analyzed", len(amenity_impacts))

with col2:
    st.metric("Recommended", len(recommended), help="Positive impact with statistical significance")

with col3:
    st.metric("Inconclusive", len(inconclusive), help="Need more data to determine impact")

with col4:
    st.metric("Not Recommended", len(not_recommended), help="No positive impact detected")


# ---------------------------------------------------------------------
# IMPACT VISUALIZATION
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Revenue Impact by Amenity")

if len(amenity_impacts) > 0:
    # Sort by revenue impact
    chart_data = amenity_impacts.sort_values('revenue_impact', ascending=True).copy()
    
    # Color based on significance and direction
    def get_color(row):
        if row['revenue_impact'] > 0 and row['significant']:
            return '#00cc96'  # Green - recommended
        elif row['revenue_impact'] > 0:
            return '#ffa15a'  # Orange - positive but uncertain
        elif row['significant']:
            return '#ef553b'  # Red - not recommended
        else:
            return '#636efa'  # Blue - inconclusive
    
    chart_data['color'] = chart_data.apply(get_color, axis=1)
    
    # Create horizontal bar chart with more height
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=chart_data['amenity'],
        x=chart_data['revenue_impact'],
        orientation='h',
        marker_color=chart_data['color'],
        text=chart_data['revenue_impact'].apply(lambda x: f"${x:+,.0f}"),
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Calculate dynamic height based on number of amenities
    chart_height = max(400, len(chart_data) * 45)
    
    fig.update_layout(
        title='Annual Revenue Impact by Amenity',
        xaxis_title='Revenue Impact ($/year)',
        yaxis_title='',
        height=chart_height,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        margin=dict(l=120, r=80, t=50, b=50)  # More margin for labels
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
    with leg_col1:
        st.markdown("🟢 **Recommended**")
    with leg_col2:
        st.markdown("🟠 **Uncertain**")
    with leg_col3:
        st.markdown("🔴 **Not Recommended**")
    with leg_col4:
        st.markdown("🔵 **Inconclusive**")
    
    # Explanation for negative impacts
    if len(not_recommended) > 0 or len(chart_data[chart_data['revenue_impact'] < 0]) > 0:
        with st.expander("Why do some amenities show negative impact?"):
            st.markdown("""
            **This doesn't necessarily mean the amenity hurts revenue.** Negative results can occur when:
            
            1. **Uncontrolled variables**: Our model controls for bedrooms, bathrooms, capacity, and neighborhood, 
               but other factors (property age, condition, exact location within neighborhood) aren't captured.
            
            2. **Sample concentration**: If most properties with outdoor kitchens are in one specific area 
               that happens to underperform, the model may attribute that area's performance to the amenity.
            
            3. **Guest type differences**: Some amenities may attract different guest types with different 
               booking patterns (e.g., longer stays at lower nightly rates).
            
            4. **Statistical noise**: With smaller sample sizes, random variation can produce misleading results.
            
            **Our recommendation**: Treat negative results as "inconclusive" rather than definitive evidence 
            that the amenity hurts revenue. Focus on amenities with positive, statistically significant impacts.
            """)


# ---------------------------------------------------------------------
# DETAILED RESULTS TABLE
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Detailed Results")

if len(amenity_impacts) > 0:
    # Create display dataframe
    display_df = amenity_impacts.copy()
    
    # Add recommendation column - shorter text to prevent cutoff
    def get_recommendation(row):
        if row['revenue_impact'] > 0 and row['p_value'] < 0.01:
            return "✅ Strong"
        elif row['revenue_impact'] > 0 and row['p_value'] < 0.05:
            return "✅ Yes"
        elif row['revenue_impact'] > 0:
            return "⚠️ Maybe"
        elif row['p_value'] < 0.05:
            return "❌ No"
        else:
            return "❓ Unclear"
    
    display_df['Recommendation'] = display_df.apply(get_recommendation, axis=1)

    # Format columns
    display_df['revenue_impact'] = display_df['revenue_impact'].round(0).astype(int)
    display_df['pct_lift'] = display_df['pct_lift'].round(1)
    display_df['n_matched'] = display_df['n_matched'].astype(int)
    
    # Select and rename columns - fewer columns to prevent scrolling
    table_df = display_df[[
        'amenity', 'revenue_impact', 'pct_lift', 'n_matched', 'Recommendation'
    ]].copy()
    
    table_df.columns = [
        'Amenity', 'Revenue Impact ($)', 'Lift (%)', 'Sample Size', 'Recommendation'
    ]
    
    # Sort by revenue impact
    table_df = table_df.sort_values('Revenue Impact ($)', ascending=False)
    
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "Revenue Impact ($)": st.column_config.NumberColumn(
                "Revenue Impact ($)",
                format="$%d",
                width="medium"
            ),
            "Lift (%)": st.column_config.NumberColumn(
                "Lift (%)",
                format="%.1f%%",
                width="small"
            ),
            "Sample Size": st.column_config.NumberColumn(
                "Sample Size",
                format="%d",
                width="small"
            ),
            "Amenity": st.column_config.TextColumn(
                "Amenity",
                width="medium"
            ),
            "Recommendation": st.column_config.TextColumn(
                "Recommendation",
                width="large"
            )
        }
    )
    
    st.markdown("---")
    
    # Recommendation guide - more compact
    st.markdown("**Recommendation Guide:**")
    
    guide_col1, guide_col2 = st.columns(2)
    
    with guide_col1:
        st.markdown("""
        - ✅ **Strong** — High confidence positive impact
        - ✅ **Yes** — Good evidence of benefit
        - ⚠️ **Maybe** — Positive but uncertain
        """)
    
    with guide_col2:
        st.markdown("""
        - ❌ **No** — No positive impact detected
        - ❓ **Unclear** — Insufficient data
        """)

# ---------------------------------------------------------------------
# INVESTMENT ANALYSIS
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Investment Analysis")
st.markdown("Should you add an amenity? Here's the projected ROI.")

if len(recommended) > 0:
    # Installation cost estimates
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
    
    # Calculate ROI for recommended amenities
    roi_data = []
    
    for _, row in recommended.iterrows():
        amenity_name = row['amenity']
        install_cost = install_costs.get(amenity_name, 5000)
        annual_benefit = int(round(row['revenue_impact']))  # Force integer
        
        if annual_benefit > 0:
            payback_years = round(install_cost / annual_benefit, 1)
            five_year_return = int((annual_benefit * 5) - install_cost)
            five_year_roi = int(round((five_year_return / install_cost) * 100))
        else:
            payback_years = 999
            five_year_return = -install_cost
            five_year_roi = -100
        
        # Only include if positive ROI
        if five_year_roi > 0:
            roi_data.append({
                'Amenity': amenity_name,
                'Install Cost': install_cost,
                'Annual Benefit': annual_benefit,
                'Payback (Years)': payback_years,
                '5-Year Return': five_year_return,
                '5-Year ROI': five_year_roi
            })
    
    if len(roi_data) > 0:
        roi_df = pd.DataFrame(roi_data)
        roi_df = roi_df.sort_values('5-Year ROI', ascending=False)
        
        # Display as chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=roi_df['Amenity'],
            y=roi_df['5-Year ROI'],
            text=roi_df['5-Year ROI'].apply(lambda x: f"{x}%"),
            textposition='inside',
            textfont=dict(size=14, color='white'),
            marker_color='#00cc96'
        ))
        
        fig.update_layout(
            title='5-Year ROI by Amenity (Recommended Only)',
            xaxis_title='',
            yaxis_title='5-Year ROI (%)',
            height=400,
            yaxis=dict(range=[0, max(roi_df['5-Year ROI']) * 1.15])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Format table for display
        display_roi = roi_df.copy()
        display_roi['Install Cost'] = display_roi['Install Cost'].apply(lambda x: f"${x:,}")
        display_roi['Annual Benefit'] = display_roi['Annual Benefit'].apply(lambda x: f"${x:,}")
        display_roi['Payback (Years)'] = display_roi['Payback (Years)'].apply(lambda x: f"{x:.1f}")
        display_roi['5-Year Return'] = display_roi['5-Year Return'].apply(lambda x: f"${x:,}")
        display_roi['5-Year ROI'] = display_roi['5-Year ROI'].apply(lambda x: f"{x}%")
        
        st.dataframe(display_roi, use_container_width=True, hide_index=True)
        
        st.caption("*Installation costs are estimates. Get actual quotes for your situation.*")
    else:
        st.info("No amenities with positive 5-year ROI found.")

else:
    st.info("No amenities met the threshold for recommendation based on current data.")

# ---------------------------------------------------------------------
# METHODOLOGY LINK
# ---------------------------------------------------------------------

st.markdown("---")
st.info("📊 **Want to understand how we calculate these impacts?** Visit the **Methodology** page for a full explanation of Propensity Score Matching.")