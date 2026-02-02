"""
Methodology Page
================

Explain the data sources, statistical methods, and assumptions
used throughout the application.
"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_listings_with_amenities, load_mls_active


st.title("📖 Methodology")
st.markdown("How we analyze short-term rental investments.")


# ---------------------------------------------------------------------
# DATA SOURCES
# ---------------------------------------------------------------------

st.header("1. Data Sources")

st.subheader("Airbnb Market Data")

airbnb = load_listings_with_amenities()
mls = load_mls_active()

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Source:** [Inside Airbnb](http://insideairbnb.com/)
    
    **Coverage:** Las Vegas metropolitan area
    
    **Records:** {len(airbnb):,} active listings
    
    **Data includes:**
    - Nightly price
    - Bedrooms, bathrooms, capacity
    - Location (neighborhood, coordinates)
    - Amenities list
    - Calendar availability (365 days)
    - Review scores and counts
    """)

with col2:
    st.markdown(f"""
    **MLS Active Listings**
    
    **Source:** Las Vegas MLS
    
    **Records:** {len(mls):,} homes for sale
    
    **Data includes:**
    - Listing price
    - Bedrooms, bathrooms
    - Square footage
    - Year built
    - Pool/spa
    - Days on market
    - Zip code
    """)

st.markdown("---")


# ---------------------------------------------------------------------
# REVENUE CALCULATION
# ---------------------------------------------------------------------

st.header("2. Revenue Calculation")

st.subheader("How We Estimate Annual Revenue")

st.markdown("""
Annual revenue is calculated from two components:
""")

st.latex(r"\text{Annual Revenue} = \text{Nightly Rate} \times \text{Occupancy Rate} \times 365")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Nightly Rate (ADR)**
    
    The listed price per night from each Airbnb listing.
    
    We filter out outliers:
    - Minimum: $10/night
    - Maximum: $1,000/night
    
    This removes data entry errors and ultra-luxury properties that skew averages.
    """)

with col2:
    st.markdown("""
    **Occupancy Rate**
    
    Calculated from the Airbnb calendar data:
    
    - Calendar shows 365 days of availability
    - Days marked "unavailable" = booked
    - Occupancy = Unavailable days ÷ 365
    
    *Note: Some unavailable days may be owner blocks, not bookings. This may slightly overestimate occupancy.*
    """)

st.subheader("Why We Use Percentiles, Not Averages")

st.warning("""
**The problem with averages:** Poorly managed properties drag down the mean. A property with bad photos, 
no pricing optimization, and slow responses will underperform — but that's not your competition.
""")

st.markdown("""
We report three performance levels:

| Level | Percentile | Meaning |
|-------|------------|---------|
| **Average Operator** | 50th | Median performance — half do better, half do worse |
| **Good Operator** | 75th | Top 25% — solid execution on the basics |
| **Excellent Operator** | 90th | Top 10% — optimized pricing, great photos, superhost status |

**Our recommendation:** Budget conservatively at the 50th percentile, target the 75th, and treat 90th as upside potential.
""")

st.markdown("---")


# ---------------------------------------------------------------------
# PROPENSITY SCORE MATCHING
# ---------------------------------------------------------------------

st.header("3. Causal Analysis: Propensity Score Matching")

st.subheader("The Problem with Simple Comparisons")

st.markdown("""
If we compare properties with pools to those without, we get biased results:
""")

col1, col2 = st.columns(2)

with col1:
    st.error("""
    **Properties WITH pools tend to have:**
    - More bedrooms
    - Better neighborhoods
    - Higher-end finishes
    - More amenities overall
    """)

with col2:
    st.markdown("""
    **The bias:**
    
    A simple comparison attributes ALL of these differences to the pool.
    
    If pool properties earn $15K more, how much is the pool vs. the location?
    """)

st.subheader("Our Solution: Propensity Score Matching (PSM)")

st.success("""
**PSM creates fair comparisons by matching similar properties.**

Instead of comparing ALL pool properties to ALL non-pool properties, we match each pool property 
to a similar non-pool property — same bedrooms, same neighborhood, same capacity.
""")

st.markdown("### How PSM Works")

st.markdown("""
**Step 1: Calculate Propensity Scores**

For each property, we build a model that predicts: "What's the probability this property has a pool?"

The model uses:
- Number of bedrooms
- Number of bathrooms
- Guest capacity
- Neighborhood

This produces a score from 0 to 1 for each property.
""")

st.info("""
**Why we exclude nightly price:** Price is part of revenue (Revenue = Price × Occupancy). 
If a pool lets you charge $50 more per night, that's VALUE the pool provides. 
We want to measure that, not control it away.
""")

st.markdown("""
**Step 2: Match Properties**

We pair each pool property with a non-pool property that has a similar propensity score.
""")

# Example matching
example_data = pd.DataFrame({
    'Property': ['A (has pool)', 'B (no pool)'],
    'Bedrooms': [4, 4],
    'Neighborhood': ['Strip', 'Strip'],
    'Capacity': [8, 8],
    'Propensity Score': [0.65, 0.63],
    'Annual Revenue': ['$48,000', '$41,000']
})

st.dataframe(example_data, use_container_width=True, hide_index=True)

st.markdown("""
These properties are similar in every way that predicts pool ownership. 
The main difference is the pool itself.

**Step 3: Calculate Impact**

We compare revenue between all matched pairs:
- Pool property: $48,000
- Matched non-pool: $41,000
- **Difference: $7,000**

Average this across hundreds of matched pairs = our causal estimate.
""")

st.subheader("Statistical Confidence")

st.markdown("""
We use statistical tests to measure certainty:

| Confidence Level | P-Value | Interpretation |
|-----------------|---------|----------------|
| Very High | < 0.01 | 99%+ certain the effect is real |
| High | < 0.05 | 95%+ certain the effect is real |
| Moderate | < 0.10 | 90%+ certain, some doubt remains |
| Low | ≥ 0.10 | Could be random noise |

**We only recommend amenities with p < 0.05 (High or Very High confidence).**
""")

st.subheader("Limitations of PSM")

st.warning("""
**What we CAN'T control for:**
- Property condition (updated vs. dated)
- Listing quality (photos, description)
- Exact location within neighborhood
- Operator skill (pricing, responsiveness)

**This means:** Our estimates capture the average effect. Your results will vary based on execution.
""")

st.markdown("---")


# ---------------------------------------------------------------------
# ROI CALCULATION
# ---------------------------------------------------------------------

st.header("4. ROI Calculation")

st.subheader("Cash-on-Cash Return")

st.markdown("""
The primary ROI metric for real estate investors:
""")

st.latex(r"\text{Cash-on-Cash ROI} = \frac{\text{Annual Cash Flow}}{\text{Down Payment}} \times 100")

st.markdown("""
**Example:**
- Down payment: $100,000
- Annual cash flow: $15,000
- Cash-on-Cash ROI: 15%
""")

st.subheader("Cash Flow Calculation")

st.markdown("""
```
  Gross Revenue (ADR × Occupancy × 365)
- Operating Expenses
  ─────────────────────────────────────
= Net Operating Income (NOI)
- Debt Service (Mortgage P&I)
  ─────────────────────────────────────
= Cash Flow
```
""")

st.subheader("Operating Expenses")

expenses_data = pd.DataFrame({
    'Category': ['Property Tax', 'Insurance', 'Utilities', 'Cleaning', 'Supplies', 'Maintenance', 'Management'],
    'Type': ['Fixed', 'Fixed', 'Fixed', 'Variable', 'Fixed', 'Variable', 'Variable'],
    'Typical Range': ['0.5-1% of value', '$2,000-4,000/yr', '$200-400/mo', '$100-200/turn', '$50-150/mo', '3-5% of revenue', '0-25% of revenue'],
    'Notes': [
        'Nevada has low property taxes',
        'Higher for STR coverage',
        'Electric, gas, water, internet, trash',
        'Depends on property size',
        'Toiletries, linens, consumables',
        'Reserve for repairs',
        '0% if self-managed'
    ]
})

st.dataframe(expenses_data, use_container_width=True, hide_index=True)

st.subheader("Mortgage Calculation")

st.markdown("""
We use standard amortization:
""")

st.latex(r"M = P \times \frac{r(1+r)^n}{(1+r)^n - 1}")

st.markdown("""
Where:
- M = Monthly payment
- P = Principal (loan amount)
- r = Monthly interest rate (annual rate ÷ 12)
- n = Number of payments (years × 12)
""")

st.markdown("---")


# ---------------------------------------------------------------------
# ASSUMPTIONS & CAVEATS
# ---------------------------------------------------------------------

st.header("5. Assumptions & Caveats")

st.subheader("Key Assumptions")

st.markdown("""
| Assumption | Our Default | Reality |
|------------|-------------|---------|
| Occupancy source | Calendar "unavailable" days | May include owner blocks |
| Revenue timing | Even throughout year | Seasonal variation exists |
| Expense estimates | Industry averages | Property-specific |
| Appreciation | 3% annually | Market dependent |
| Revenue growth | 2% annually | Not guaranteed |
""")

st.subheader("What This Analysis Does NOT Account For")

st.error("""
- **Regulatory risk**: STR regulations can change
- **Market saturation**: More supply could reduce rates
- **Startup costs**: Furniture, setup, photography
- **Vacancy during setup**: Time to first booking
- **Your time**: Self-management has opportunity cost
- **Financing variations**: Points, closing costs, PMI
""")

st.subheader("Our Recommendation")

st.success("""
**Use this tool for:**
- Initial screening of properties
- Comparing relative opportunities
- Understanding market dynamics
- Identifying high-impact amenities

**Before purchasing:**
- Get actual quotes for expenses
- Verify rental regulations
- Analyze specific comps for the property
- Consult with a real estate professional
- Run your own financial projections
""")

st.markdown("---")


# ---------------------------------------------------------------------
# DATA FRESHNESS
# ---------------------------------------------------------------------

st.header("6. Data Freshness")

st.markdown("""
| Data Source | Update Frequency | Last Updated |
|-------------|------------------|--------------|
| Airbnb listings | Quarterly | See Inside Airbnb |
| Airbnb calendar | Quarterly | See Inside Airbnb |
| MLS active listings | Manual refresh | User uploaded |
| Hotel rates | On-demand via API | Per session |

*For the most current analysis, refresh your MLS data periodically.*
""")

st.markdown("---")


# ---------------------------------------------------------------------
# CONTACT / FEEDBACK
# ---------------------------------------------------------------------

st.header("7. Questions or Feedback?")

st.markdown("""
This tool was built to help investors make data-driven decisions.

If you have questions, suggestions, or find issues:
- Open an issue on [GitHub](https://github.com/yourusername/str-analyzer)
- Or reach out directly

**Built with:** Python, Streamlit, Pandas, Plotly, Scikit-learn
""")