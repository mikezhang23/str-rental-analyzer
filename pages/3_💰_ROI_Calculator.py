"""
ROI Calculator Page
===================

Detailed investment analysis with customizable inputs.
Users input property details and assumptions, get comprehensive ROI projections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import load_listings_with_amenities, load_mls_active
from utils.stats import get_all_amenity_impacts


st.title("💰 ROI Calculator")
st.markdown("Model your investment returns with detailed assumptions.")


# ---------------------------------------------------------------------
# LOAD DATA FOR BENCHMARKS
# ---------------------------------------------------------------------

airbnb = load_listings_with_amenities()
amenity_impacts = get_all_amenity_impacts(airbnb)


# ---------------------------------------------------------------------
# SIDEBAR: PROPERTY INPUTS
# ---------------------------------------------------------------------

st.sidebar.header("Property Details")

# Purchase details
st.sidebar.subheader("Purchase")

purchase_price = st.sidebar.number_input(
    "Purchase Price ($)",
    min_value=100000,
    max_value=2000000,
    value=400000,
    step=10000,
    format="%d"
)

down_payment_pct = st.sidebar.slider(
    "Down Payment (%)",
    min_value=0,
    max_value=100,
    value=25,
    step=5
)

interest_rate = st.sidebar.slider(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.25
)

loan_term = st.sidebar.selectbox(
    "Loan Term (years)",
    options=[15, 20, 30],
    index=2
)

# Property characteristics
st.sidebar.subheader("Property")

bedrooms = st.sidebar.selectbox(
    "Bedrooms",
    options=[1, 2, 3, 4, 5, 6],
    index=2  # Default to 3
)

bathrooms = st.sidebar.selectbox(
    "Bathrooms",
    options=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6],
    index=3  # Default to 2.5
)

location_quality = st.sidebar.selectbox(
    "Location Quality",
    options=["Poor", "Below Average", "Average", "Good", "Excellent"],
    index=2
)

# Amenities
st.sidebar.subheader("Amenities")

has_pool = st.sidebar.checkbox("Pool", value=False)
has_hot_tub = st.sidebar.checkbox("Hot Tub", value=False)
has_game_room = st.sidebar.checkbox("Game Room", value=False)


# ---------------------------------------------------------------------
# SIDEBAR: OPERATING ASSUMPTIONS
# ---------------------------------------------------------------------

st.sidebar.header("Operating Assumptions")

# Revenue assumptions
st.sidebar.subheader("Revenue")

operator_level = st.sidebar.selectbox(
    "Operator Skill Level",
    options=["Average (50th percentile)", "Good (75th percentile)", "Excellent (90th percentile)"],
    index=1
)

# Map to percentile
operator_percentile = {
    "Average (50th percentile)": 0.50,
    "Good (75th percentile)": 0.75,
    "Excellent (90th percentile)": 0.90
}[operator_level]

# Expense assumptions
st.sidebar.subheader("Expenses")

property_tax_rate = st.sidebar.slider(
    "Property Tax Rate (%)",
    min_value=0.0,
    max_value=3.0,
    value=0.7,  # Nevada is low
    step=0.1
)

insurance_annual = st.sidebar.number_input(
    "Annual Insurance ($)",
    min_value=0,
    max_value=10000,
    value=2400,
    step=100
)

hoa_monthly = st.sidebar.number_input(
    "HOA Monthly ($)",
    min_value=0,
    max_value=1000,
    value=0,
    step=25
)

management_pct = st.sidebar.slider(
    "Management Fee (%)",
    min_value=0,
    max_value=30,
    value=0,  # Self-managed default
    step=5,
    help="0% if self-managed, 20-25% for full-service management"
)

cleaning_per_turn = st.sidebar.number_input(
    "Cleaning Cost (per turnover)",
    min_value=0,
    max_value=500,
    value=150,
    step=25
)

utilities_monthly = st.sidebar.number_input(
    "Utilities Monthly ($)",
    min_value=0,
    max_value=1000,
    value=300,
    step=25,
    help="Electric, gas, water, internet, trash"
)

maintenance_pct = st.sidebar.slider(
    "Maintenance Reserve (%)",
    min_value=0,
    max_value=10,
    value=5,
    step=1,
    help="Percentage of revenue set aside for repairs"
)

supplies_monthly = st.sidebar.number_input(
    "Supplies Monthly ($)",
    min_value=0,
    max_value=500,
    value=100,
    step=25,
    help="Toiletries, linens replacement, consumables"
)


# ---------------------------------------------------------------------
# CALCULATIONS
# ---------------------------------------------------------------------

# --- Mortgage Calculation ---
down_payment = purchase_price * (down_payment_pct / 100)
loan_amount = purchase_price - down_payment
monthly_rate = (interest_rate / 100) / 12
num_payments = loan_term * 12

if monthly_rate > 0:
    monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
else:
    monthly_mortgage = loan_amount / num_payments if num_payments > 0 else 0

annual_mortgage = monthly_mortgage * 12

# --- Revenue Projection ---
# Get baseline revenue for this bedroom count
bedroom_data = airbnb[airbnb['bedrooms'] == bedrooms]

if len(bedroom_data) > 0:
    base_revenue = bedroom_data['annual_revenue'].quantile(operator_percentile)
    median_revenue = bedroom_data['annual_revenue'].median()
    avg_occupancy = bedroom_data['occupancy_rate'].median()
    avg_adr = bedroom_data['price_clean'].median()
else:
    base_revenue = 30000  # Fallback
    median_revenue = 25000
    avg_occupancy = 0.45
    avg_adr = 150

# Location adjustment
location_multipliers = {
    "Poor": 0.60,
    "Below Average": 0.80,
    "Average": 1.00,
    "Good": 1.15,
    "Excellent": 1.30
}
location_mult = location_multipliers[location_quality]

# Amenity adjustments (from causal analysis)
amenity_boost = 0

if has_pool:
    pool_impact = amenity_impacts[amenity_impacts['amenity'] == 'Pool']
    if len(pool_impact) > 0 and pool_impact.iloc[0]['revenue_impact'] > 0:
        amenity_boost += pool_impact.iloc[0]['revenue_impact']
    else:
        amenity_boost += 5000  # Fallback estimate

if has_hot_tub:
    hot_tub_impact = amenity_impacts[amenity_impacts['amenity'] == 'Hot Tub']
    if len(hot_tub_impact) > 0 and hot_tub_impact.iloc[0]['revenue_impact'] > 0:
        amenity_boost += hot_tub_impact.iloc[0]['revenue_impact']
    else:
        amenity_boost += 2000

if has_game_room:
    game_impact = amenity_impacts[amenity_impacts['amenity'] == 'Game Room']
    if len(game_impact) > 0 and game_impact.iloc[0]['revenue_impact'] > 0:
        amenity_boost += game_impact.iloc[0]['revenue_impact']
    else:
        amenity_boost += 1500

# Final revenue projection
projected_revenue = (base_revenue * location_mult) + amenity_boost

# Estimate occupancy and ADR
projected_occupancy = avg_occupancy * location_mult
projected_occupancy = min(projected_occupancy, 0.85)  # Cap at 85%
projected_adr = projected_revenue / (projected_occupancy * 365)

# Estimate number of turnovers (average stay = 3 nights)
avg_stay_nights = 3
turnovers_per_year = (projected_occupancy * 365) / avg_stay_nights

# --- Expense Calculations ---
property_tax = purchase_price * (property_tax_rate / 100)
insurance = insurance_annual
hoa = hoa_monthly * 12
management_fee = projected_revenue * (management_pct / 100)
cleaning_total = cleaning_per_turn * turnovers_per_year
utilities = utilities_monthly * 12
maintenance = projected_revenue * (maintenance_pct / 100)
supplies = supplies_monthly * 12

total_operating_expenses = (
    property_tax + insurance + hoa + management_fee + 
    cleaning_total + utilities + maintenance + supplies
)

total_expenses = annual_mortgage + total_operating_expenses

# --- Final Metrics ---
net_operating_income = projected_revenue - total_operating_expenses
cash_flow = projected_revenue - total_expenses
cash_on_cash_roi = (cash_flow / down_payment) * 100 if down_payment > 0 else 0
cap_rate = (net_operating_income / purchase_price) * 100


# ---------------------------------------------------------------------
# MAIN CONTENT: RESULTS
# ---------------------------------------------------------------------

# Key Metrics Row
st.subheader("Investment Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Annual Cash Flow",
        f"${cash_flow:,.0f}",
        delta=f"${cash_flow/12:,.0f}/month"
    )

with col2:
    st.metric(
        "Cash-on-Cash ROI",
        f"{cash_on_cash_roi:.1f}%",
        help="Annual cash flow ÷ down payment"
    )

with col3:
    st.metric(
        "Cap Rate",
        f"{cap_rate:.1f}%",
        help="NOI ÷ purchase price (property return independent of financing)"
    )

with col4:
    st.metric(
        "Monthly Cash Flow",
        f"${cash_flow/12:,.0f}"
    )


# ---------------------------------------------------------------------
# REVENUE BREAKDOWN
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Revenue Projection")

rev_col1, rev_col2, rev_col3 = st.columns(3)

with rev_col1:
    st.metric("Projected Annual Revenue", f"${projected_revenue:,.0f}")

with rev_col2:
    st.metric("Projected ADR", f"${projected_adr:,.0f}")

with rev_col3:
    st.metric("Projected Occupancy", f"{projected_occupancy*100:.0f}%")

# Revenue breakdown
with st.expander("Revenue Calculation Details"):
    st.markdown(f"""
    | Component | Value |
    |-----------|-------|
    | Base revenue ({bedrooms} bed, {operator_level.split()[0].lower()} operator) | ${base_revenue:,.0f} |
    | Location adjustment ({location_quality}) | ×{location_mult:.2f} |
    | Amenity boost | +${amenity_boost:,.0f} |
    | **Projected Revenue** | **${projected_revenue:,.0f}** |
    """)
    
    st.caption(f"Based on {len(bedroom_data):,} comparable {bedrooms}-bedroom Airbnb listings in Las Vegas")


# ---------------------------------------------------------------------
# EXPENSE BREAKDOWN
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Expense Breakdown")

# Create expense dataframe for chart
expenses_data = pd.DataFrame({
    'Category': ['Mortgage (P&I)', 'Property Tax', 'Insurance', 'HOA', 
                 'Management', 'Cleaning', 'Utilities', 'Maintenance', 'Supplies'],
    'Annual': [annual_mortgage, property_tax, insurance, hoa,
               management_fee, cleaning_total, utilities, maintenance, supplies],
    'Type': ['Debt Service', 'Fixed', 'Fixed', 'Fixed',
             'Variable', 'Variable', 'Fixed', 'Variable', 'Fixed']
})

expenses_data['Monthly'] = expenses_data['Annual'] / 12

# Two columns: chart and table
chart_col, table_col = st.columns([1, 1])

with chart_col:
    # Pie chart of expenses
    fig = px.pie(
        expenses_data,
        values='Annual',
        names='Category',
        title='Annual Expenses by Category',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with table_col:
    # Expense table
    display_expenses = expenses_data[['Category', 'Monthly', 'Annual']].copy()
    display_expenses['Monthly'] = display_expenses['Monthly'].apply(lambda x: f"${x:,.0f}")
    display_expenses['Annual'] = display_expenses['Annual'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_expenses, use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    **Totals:**
    - Operating Expenses: **${total_operating_expenses:,.0f}/year**
    - Debt Service: **${annual_mortgage:,.0f}/year**
    - Total Expenses: **${total_expenses:,.0f}/year**
    """)


# ---------------------------------------------------------------------
# CASH FLOW SUMMARY
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Cash Flow Summary")

# Visual cash flow breakdown
cash_flow_data = {
    'Item': ['Gross Revenue', 'Operating Expenses', 'Net Operating Income (NOI)', 
             'Debt Service', 'Cash Flow'],
    'Amount': [projected_revenue, -total_operating_expenses, net_operating_income,
               -annual_mortgage, cash_flow]
}

cf_df = pd.DataFrame(cash_flow_data)

# Waterfall chart
fig = go.Figure(go.Waterfall(
    name="Cash Flow",
    orientation="v",
    measure=["absolute", "relative", "total", "relative", "total"],
    x=cf_df['Item'],
    y=[projected_revenue, -total_operating_expenses, 0, -annual_mortgage, 0],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    decreasing={"marker": {"color": "#ef553b"}},
    increasing={"marker": {"color": "#00cc96"}},
    totals={"marker": {"color": "#636efa"}}
))

fig.update_layout(
    title="Annual Cash Flow Waterfall",
    showlegend=False,
    height=400
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Sensitivity Analysis")
st.caption("How changes in key assumptions affect your returns")

sens_col1, sens_col2 = st.columns(2)

with sens_col1:
    st.markdown("**Occupancy Sensitivity**")
    
    occupancy_scenarios = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    occupancy_results = []
    
    for occ in occupancy_scenarios:
        scenario_revenue = projected_adr * occ * 365
        scenario_turnovers = (occ * 365) / avg_stay_nights
        scenario_cleaning = cleaning_per_turn * scenario_turnovers
        scenario_mgmt = scenario_revenue * (management_pct / 100)
        scenario_maint = scenario_revenue * (maintenance_pct / 100)
        
        scenario_op_exp = (
            property_tax + insurance + hoa + scenario_mgmt +
            scenario_cleaning + utilities + scenario_maint + supplies
        )
        
        scenario_cash_flow = scenario_revenue - scenario_op_exp - annual_mortgage
        
        occupancy_results.append({
            'Occupancy': f"{occ*100:.0f}%",
            'Revenue': f"${scenario_revenue:,.0f}",
            'Cash Flow': f"${scenario_cash_flow:,.0f}",
            'Monthly': f"${scenario_cash_flow/12:,.0f}"
        })
    
    occ_df = pd.DataFrame(occupancy_results)
    st.dataframe(occ_df, use_container_width=True, hide_index=True)

with sens_col2:
    st.markdown("**Interest Rate Sensitivity**")
    
    rate_scenarios = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    rate_results = []
    
    for rate in rate_scenarios:
        scenario_monthly_rate = (rate / 100) / 12
        if scenario_monthly_rate > 0:
            scenario_mortgage = loan_amount * (scenario_monthly_rate * (1 + scenario_monthly_rate)**num_payments) / ((1 + scenario_monthly_rate)**num_payments - 1)
        else:
            scenario_mortgage = loan_amount / num_payments
        
        scenario_annual_mortgage = scenario_mortgage * 12
        scenario_cash_flow = projected_revenue - total_operating_expenses - scenario_annual_mortgage
        
        rate_results.append({
            'Rate': f"{rate:.1f}%",
            'Monthly P&I': f"${scenario_mortgage:,.0f}",
            'Cash Flow': f"${scenario_cash_flow:,.0f}",
            'CoC ROI': f"{(scenario_cash_flow/down_payment)*100:.1f}%"
        })
    
    rate_df = pd.DataFrame(rate_results)
    st.dataframe(rate_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# 5-YEAR PROJECTION
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("5-Year Projection")

# Assumptions for growth
appreciation_rate = 0.03  # 3% annual
revenue_growth_rate = 0.02  # 2% annual

years = list(range(0, 6))
cumulative_cash_flow = 0
projection_data = []

for year in years:
    if year == 0:
        projection_data.append({
            'Year': 'Purchase',
            'Property Value': purchase_price,
            'Annual Cash Flow': 0,
            'Cumulative Cash Flow': -down_payment,
            'Equity': down_payment
        })
    else:
        year_property_value = purchase_price * ((1 + appreciation_rate) ** year)
        year_revenue = projected_revenue * ((1 + revenue_growth_rate) ** year)
        
        # Simplified: assume expenses grow at same rate
        year_expenses = total_expenses * ((1 + revenue_growth_rate) ** year)
        year_cash_flow = year_revenue - year_expenses
        
        cumulative_cash_flow += year_cash_flow
        
        # Rough equity calculation (simplified)
        # In reality would need amortization schedule
        principal_paid_estimate = annual_mortgage * 0.3 * year  # Rough estimate
        year_equity = down_payment + (year_property_value - purchase_price) + principal_paid_estimate
        
        projection_data.append({
            'Year': f'Year {year}',
            'Property Value': year_property_value,
            'Annual Cash Flow': year_cash_flow,
            'Cumulative Cash Flow': cumulative_cash_flow - down_payment,
            'Equity': year_equity
        })

proj_df = pd.DataFrame(projection_data)

# Display table
display_proj = proj_df.copy()
display_proj['Property Value'] = display_proj['Property Value'].apply(lambda x: f"${x:,.0f}")
display_proj['Annual Cash Flow'] = display_proj['Annual Cash Flow'].apply(lambda x: f"${x:,.0f}")
display_proj['Cumulative Cash Flow'] = display_proj['Cumulative Cash Flow'].apply(lambda x: f"${x:,.0f}")
display_proj['Equity'] = display_proj['Equity'].apply(lambda x: f"${x:,.0f}")

st.dataframe(display_proj, use_container_width=True, hide_index=True)

# Chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=proj_df['Year'],
    y=proj_df['Cumulative Cash Flow'],
    name='Cumulative Cash Flow',
    marker_color='#00cc96'
))

fig.add_trace(go.Scatter(
    x=proj_df['Year'],
    y=proj_df['Equity'],
    name='Total Equity',
    mode='lines+markers',
    marker_color='#636efa',
    yaxis='y2'
))

fig.update_layout(
    title='5-Year Investment Growth',
    yaxis=dict(title='Cumulative Cash Flow ($)'),
    yaxis2=dict(title='Equity ($)', overlaying='y', side='right'),
    legend=dict(x=0.01, y=0.99),
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Assumes {appreciation_rate*100:.0f}% annual appreciation and {revenue_growth_rate*100:.0f}% revenue growth")


# ---------------------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------------------

st.markdown("---")
st.warning("""
**Disclaimer:** These projections are estimates based on market data and assumptions you've provided. 
Actual results will vary based on property-specific factors, market conditions, and operator performance. 
This is not financial advice. Consult with real estate and tax professionals before making investment decisions.
""")