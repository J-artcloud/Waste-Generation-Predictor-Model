"""
============================================================
  waste_app.py — Waste Generation Predictor
============================================================

HOW TO RUN THIS APP:
  1. Make sure these files are in the same folder as this script:
       - waste_model.pkl
       - label_encoder.pkl
       - project4_waste_generation.csv
  2. Open your terminal in that folder
  3. Run: streamlit run waste_app.py
  4. A browser window will open automatically at http://localhost:8501
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
# streamlit is the library that turns this Python script into a web app
# Every st.something() call creates something visible on the web page
import streamlit as st

# pandas is used to load the CSV data and build DataFrames
import pandas as pd

# numpy gives us math tools (used for max/round operations)
import numpy as np

# matplotlib is used to draw charts displayed in the app
import matplotlib.pyplot as plt

# joblib loads the trained model we saved from the notebook
import joblib

# time is a built-in Python module — we use time.sleep() to pause for 1 second
# This creates the "processing" loading effect when the button is clicked
import time

# datetime helps us calculate real calendar dates for the collection schedule
# date.today() gives today's date, timedelta lets us add/subtract days
from datetime import date, timedelta

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# st.set_page_config() must be the FIRST streamlit command in the file
# page_title = text shown on the browser tab
# page_icon  = emoji shown on the browser tab
# layout     = 'wide' uses the full browser width instead of a narrow center column
st.set_page_config(
    page_title="Waste Predictor",
    page_icon="🗑️",
    layout="wide"
)


import streamlit as st
import base64

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.98)),
        url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("background1.jpeg")


# ── LOAD MODEL AND DATA ───────────────────────────────────────────────────────

# @st.cache_resource tells Streamlit to load this only ONCE
# and reuse the result every time the page refreshes
# Without this, the model would reload on every click — very slow!
@st.cache_resource
def load_model():
    # joblib.load() reads the .pkl file and restores the trained model object
    model = joblib.load('waste_model.pkl')
    # Also load the LabelEncoder so we can convert neighborhood names to numbers
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

# @st.cache_data is similar but used for data (DataFrames, lists, etc.)
# It caches the result so the CSV is not re-read on every interaction
@st.cache_data
def load_data():
    # pd.read_csv() reads the CSV file into a DataFrame (like a spreadsheet)
    df = pd.read_csv('project4_waste_generation.csv')
    return df

# Try to load everything — if a file is missing, show a helpful error message
try:
    model, label_encoder = load_model()  # load trained Random Forest + encoder
    df = load_data()                     # load the original dataset
    model_loaded = True                  # flag to indicate success
except Exception as e:
    # st.error() shows a red error box on the page
    st.error(f"Could not load model or data: {e}")
    # st.info() shows a blue info box with instructions
    st.info("Make sure waste_model.pkl, label_encoder.pkl, and project4_waste_generation.csv are all in the same folder as this script.")
    # st.stop() stops the app from running any further code
    st.stop()

# ── HELPER VALUES ──────────────────────────────────────────────────────────────

# Get a sorted list of all unique neighborhood names from the dataset
# sorted() arranges them alphabetically for easy browsing in the dropdown
neighborhoods = sorted(df['Neighborhood'].unique().tolist())

# Dictionary mapping month numbers (1–12) to their full names
# Used to display "March" instead of "3" in the interface
month_names = {
    1: "January",  2: "February", 3: "March",    4: "April",
    5: "May",       6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
# st.markdown() with unsafe_allow_html=True lets us inject raw HTML and CSS
# This is how we customize colors, fonts, and layout beyond Streamlit's defaults
st.markdown("""
<style>
    /* Style for the main page title */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        padding: 10px 0;
    }
    /* Style for the subtitle below the title */
    .subtitle {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Override Streamlit's default blue primary button → make it GREEN */
    div.stButton > button[kind="primary"] {
        background-color: #1e7e34 !important;
        border-color: #1e7e34 !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        padding: 0.6rem 1rem !important;
        border-radius: 8px !important;
    }
    /* Slightly darker green when the user hovers over the button */
    div.stButton > button[kind="primary"]:hover {
        background-color: #155724 !important;
        border-color: #155724 !important;
    }
    /* Blue gradient box used to display the predicted waste number */
    .result-box {
        background: linear-gradient(135deg, #1f4e79, #2e75b6);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    /* Orange gradient box used to display the truck recommendation */
    .truck-box {
        background: linear-gradient(135deg, #833c0b, #c55a11);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    /* Green card used for each individual collection date */
    .date-card {
        background: linear-gradient(135deg, #155724, #28a745);
        color: white;
        padding: 14px 18px;
        border-radius: 10px;
        text-align: center;
        margin: 6px 0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    /* Wrapper box for the whole collection schedule section */
    .schedule-box {
        background: #f0fff4;
        border: 1px solid #28a745;
        border-radius: 12px;
        padding: 18px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── PAGE HEADER ────────────────────────────────────────────────────────────────
# Display the main title using our custom CSS class defined above
st.markdown('<div class="main-title">🗑️ Neighborhood Waste Predictor</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; font-style: italic; color: #2e7d32;">Predicting today for a sustainable tomorrow</div>', unsafe_allow_html=True)
# st.markdown("---") draws a horizontal dividing line
st.markdown("---")

# ── NAVIGATION TABS ───────────────────────────────────────────────────────────
# st.tabs() creates clickable tabs at the top of the page
# Each tab is a separate section of the app
# tab1, tab2, tab3 are context managers — we use 'with tab1:' to add content to each tab
tab1, tab2, tab3 = st.tabs(["🔮 Make a Prediction", "📊 Explore the Data", "ℹ️ How It Works"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — MAKE A PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # st.subheader() displays a medium-sized heading
    st.subheader("Enter Neighborhood Details")

    # st.write() displays text (like print() but for the web page)
    st.write("Fill in the details below and click **Predict** to see the estimated waste for that week.")

    # st.columns(2) splits the page into 2 equal side-by-side columns
    # col1 = left column, col2 = right column
    col1, col2 = st.columns(2)

    # Everything inside 'with col1:' appears in the LEFT column
    with col1:
        # st.markdown() renders Markdown text (** makes text bold)
        st.markdown("**📍 Location & Time**")

        # st.selectbox() creates a dropdown menu for selecting one option
        # options= is the list of choices shown in the dropdown
        # help= shows a tooltip when the user hovers over the widget
        selected_neighborhood = st.selectbox(
            "Select Neighborhood",
            options=neighborhoods,
            help="Choose the neighborhood you want to predict waste for"
        )

        # Another dropdown, this time for selecting the month
        # format_func= lets us display a custom label for each option
        # lambda x: takes the number x and returns the full month name
        selected_month = st.selectbox(
            "Select Month",
            options=list(month_names.keys()),          # options are 1, 2, ..., 12
            format_func=lambda x: month_names[x],      # display as "January", "February"...
            index=2,                                    # default to index 2 = March (0-indexed)
            help="Which month of the year are you predicting for?"
        )

        # st.slider() creates a draggable slider for selecting a number in a range
        # min_value and max_value set the range
        # value= sets the default starting position
        collection_freq = st.slider(
            "Collection Frequency (per week)",
            min_value=1, max_value=7, value=2,
            help="How many times per week do waste collection trucks come?"
        )

    # Everything inside 'with col2:' appears in the RIGHT column
    with col2:
        st.markdown("**👥 Neighborhood Information**")

        # st.number_input() creates a text box where users type a number
        # step= controls how much the value changes when clicking the +/- arrows
        population_density = st.number_input(
            "Population Density (persons/km²)",
            min_value=100, max_value=20000,
            value=4400, step=100,
            help="How many people live per square kilometer in this area?"
        )

        num_businesses = st.number_input(
            "Number of Businesses",
            min_value=0, max_value=5000,
            value=430, step=10,
            help="Total number of shops, offices, restaurants, etc."
        )

        # st.slider() with float step for decimal values
        avg_household = st.slider(
            "Average Household Size (people)",
            min_value=1.0, max_value=20.0, value=8.2, step=0.1,
            help="On average, how many people live in each household?"
        )

        # st.radio() creates a set of circular radio buttons (choose one)
        # horizontal=True arranges them side by side instead of stacked
        market_present = st.radio(
            "Is there a Market in this area?",
            options=[0, 1],
            format_func=lambda x: "Yes ✅" if x == 1 else "No ❌",
            horizontal=True
        )

        industrial_zone = st.radio(
            "Is it an Industrial Zone?",
            options=[0, 1],
            format_func=lambda x: "Yes ✅" if x == 1 else "No ❌",
            horizontal=True
        )

    # Draw a dividing line between input section and results
    st.markdown("---")

    # ── PREDICT BUTTON ────────────────────────────────────────────────────────
    # type="primary" triggers our CSS override which makes the button GREEN
    # use_container_width=True stretches the button across the full column width
    # The entire 'if' block below only runs when the user clicks this button


# 1. Create three columns. 
# The middle one (col2) holds the button. 
# The [2, 1, 2] ratio means the side columns are twice as wide as the center.
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # 2. Minimal CSS to just handle the centering without stretching
    st.markdown("""
        <style>
        div.stButton {
            margin:0 auto;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 3. Your button
    if st.button("🟢 Predict Waste Generation"):
        # (Your prediction logic goes here)
        



   





        # ── LOADING SPINNER ───────────────────────────────────────────────────
        # st.spinner() shows a spinning animation with a message while the
        # code inside the 'with' block is running
        # time.sleep(1) pauses execution for 1 second so the spinner is visible
        with st.spinner("⏳ Processing data... please wait"):
            time.sleep(4)  # pause 1 second to show the loading animation

            # Step 1: Convert neighborhood name to its encoded number
            # label_encoder.transform() expects a list, so we wrap the name in []
            # [0] extracts the single integer result from the returned array
            neighborhood_code = label_encoder.transform([selected_neighborhood])[0]

            # Step 2: Build the input as a single-row DataFrame
            # Column names and order must EXACTLY match what was used during training
            input_data = pd.DataFrame([{
                'Population_Density_per_km2':    population_density,
                'Number_of_Businesses':           num_businesses,
                'Avg_Household_Size':             avg_household,
                'Market_Present':                 market_present,
                'Industrial_Zone':                industrial_zone,
                'Collection_Frequency_per_Week':  collection_freq,
                'Month':                          selected_month,
                'Neighborhood_Encoded':           neighborhood_code
            }])

            # Step 3: Run prediction through the trained Random Forest model
            # .predict() passes the input through all 100 trees and returns their average
            # [0] gets the single float value from the result array
            predicted_waste = model.predict(input_data)[0]

            # Ensure the prediction is never negative (waste weight can't be below 0)
            predicted_waste = max(0, predicted_waste)

            # Step 4: Calculate how many trucks to send
            # Each truck holds 1500 kg — divide total waste by capacity
            # round() gives a whole number, max(1,...) ensures at least 1 truck always
            TRUCK_CAPACITY_KG = 1500
            trucks_needed = max(1, round(predicted_waste / TRUCK_CAPACITY_KG))

            # ── Calculate Specific Collection Dates ───────────────────────────
            # We assign real calendar dates for collection days
            # Collection days are ONLY from Tuesday to Saturday (weekdays 1–5)
            # Python weekday(): Monday=0, Tuesday=1, Wednesday=2, Thursday=3,
            #                   Friday=4, Saturday=5, Sunday=6

            # All allowed collection days are Tuesday(1) through Saturday(5)
            allowed_weekdays = [1, 2, 3, 4, 5]  # Tue, Wed, Thu, Fri, Sat

            # Pre-set spread patterns based on collection frequency (1–5 days)
            # These weekday numbers (1–5) represent Tue, Wed, Thu, Fri, Sat
            # We pick days that spread the collections as evenly as possible
            schedule_patterns = {
                1: [2],           # 1 collection → Wednesday only (mid-week)
                2: [1, 4],        # 2 collections → Tuesday and Friday
                3: [1, 3, 5],     # 3 collections → Tuesday, Thursday, Saturday
                4: [1, 2, 4, 5],  # 4 collections → Tue, Wed, Fri, Sat
                5: [1, 2, 3, 4, 5], # 5 collections → every day Tue–Sat
            }

            # If collection_freq > 5, cap it at 5 (only 5 allowed days exist)
            freq_capped = min(collection_freq, 5)

            # Get the list of target weekday numbers for this frequency
            target_weekdays = schedule_patterns[freq_capped]

            # Find today's date so we can calculate real upcoming dates
            today = date.today()

            # Find the next upcoming Tuesday (the start of our collection window)
            # today.weekday() gives today's weekday number (Mon=0 ... Sun=6)
            # We calculate how many days until next Tuesday
            days_until_tuesday = (1 - today.weekday()) % 7
            # If today IS Tuesday, days_until_tuesday = 0, so we use this week
            # If today is past Tuesday, we go to NEXT Tuesday
            if days_until_tuesday == 0 and today.weekday() == 1:
                next_tuesday = today  # today is Tuesday, use this week
            else:
                # Add the number of days to reach next Tuesday
                # If days_until_tuesday == 0 but we're not on Tuesday, move to next week
                if days_until_tuesday == 0:
                    days_until_tuesday = 7
                next_tuesday = today + timedelta(days=days_until_tuesday)

            # Build the list of actual calendar dates for collection
            # For each target weekday offset from Tuesday (0=Tue, 1=Wed, 2=Thu, 3=Fri, 4=Sat)
            # we add the appropriate number of days to next_tuesday
            collection_dates = []
            for weekday_num in target_weekdays:
                # weekday_num - 1 gives the offset from Tuesday (since Tuesday = 1)
                offset = weekday_num - 1  # e.g. Friday(4) - 1 = 3 days after Tuesday
                collection_date = next_tuesday + timedelta(days=offset)
                # Format as "Monday, 15 Jan 2025" style
                collection_dates.append(collection_date.strftime("%A"))

        # ── DISPLAY RESULTS (shown AFTER spinner disappears) ──────────────────
        st.markdown("## 🎯 Prediction Results")

        # Split the page into 3 equal columns for the result summary cards
        col_a, col_b, col_c = st.columns(3)

        # Column A: Predicted waste weight (blue gradient card)
        with col_a:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:0.95rem; opacity:0.85">Predicted Weekly Waste</div>
                <div style="font-size:2.5rem; font-weight:bold">{predicted_waste:,.0f} kg</div>
                <div style="font-size:0.85rem; opacity:0.8">{month_names[selected_month]}</div>
            </div>
            """, unsafe_allow_html=True)

        # Column B: Truck recommendation (orange gradient card)
        with col_b:
            # '🚛 ' * trucks_needed repeats the truck emoji trucks_needed times
            st.markdown(f"""
            <div class="truck-box">
                <div style="font-size:0.95rem; opacity:0.9">Trucks Recommended</div>
                <div style="font-size:2.5rem; font-weight:bold">{'🚛 ' * trucks_needed}{trucks_needed}</div>
                <div style="font-size:0.85rem; opacity:0.9">@ {TRUCK_CAPACITY_KG:,} kg capacity each</div>
            </div>
            """, unsafe_allow_html=True)

        # Column C: Compare to this neighborhood's historical average
        with col_c:
            # Filter dataset rows for the selected neighborhood, then get their average
            neighborhood_avg = df[df['Neighborhood'] == selected_neighborhood]['Weekly_Waste_Weight_kg'].mean()

            # How different is our prediction from the historical average?
            diff = predicted_waste - neighborhood_avg

            # Pick label and color: red = above average (more waste), green = below
            direction = "above" if diff > 0 else "below"
            color = "#cc0000" if diff > 0 else "#006600"

            st.markdown(f"""
            <div style="background:#f8f8f8; border:1px solid #ddd; padding:20px; border-radius:15px; text-align:center;">
                <div style="font-size:0.95rem; color:#555">vs. Historical Average</div>
                <div style="font-size:2rem; font-weight:bold; color:{color}">{abs(diff):,.0f} kg</div>
                <div style="font-size:0.85rem; color:#555">{direction} the avg ({neighborhood_avg:,.0f} kg)</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Recommendation Message ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Recommendation")

        # st.success() = green box, st.warning() = yellow/orange box
        if trucks_needed == 1:
            st.success(
                f"**{selected_neighborhood}** is predicted to generate **{predicted_waste:,.0f} kg** "
                f"in {month_names[selected_month]}. Send **1 truck** for collection."
            )
        else:
            st.warning(
                f"**{selected_neighborhood}** is predicted to generate **{predicted_waste:,.0f} kg** "
                f"in {month_names[selected_month]}. Send **{trucks_needed} trucks** for collection."
            )

        # ── Collection Schedule with Specific Dates ───────────────────────────
        st.markdown("### 🗓️ Collection Schedule")
        st.write(f"Scheduled **{len(collection_dates)} collection day(s)** for **{selected_neighborhood}** "
                 f"— dates are within Tuesday to Saturday:")

        # Display each collection date in its own green date card
        # We use columns to lay them out side by side (up to 3 per row)
        # math.ceil would help here but we keep it simple with a loop approach

        # Split dates into rows of 3 cards each using list slicing
        # For example if 5 dates: row1 = dates[0:3], row2 = dates[3:5]
        chunk_size = 3  # how many date cards per row

        # Loop through the dates in chunks of 3
        for i in range(0, len(collection_dates), chunk_size):
            # Get the current chunk (up to 3 dates)
            chunk = collection_dates[i : i + chunk_size]

            # Create as many columns as there are dates in this chunk
            cols = st.columns(len(chunk))

            # Place each date card into its column
            for col, date_str in zip(cols, chunk):
                with col:
                    # Each date gets a styled green card via our CSS class
                    st.markdown(
                        f'<div class="date-card">🚛 {date_str}</div>',
                        unsafe_allow_html=True
                    )

        # Final note about the schedule window
        st.caption(
            f"📌 Collection days for upcoming week"
            f"Collection days are Tuesday to Saturday only."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EXPLORE THE DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Explore the Dataset")
    st.write("Charts and summaries of the waste data used to train the model.")

    # ── Summary Metrics Row ───────────────────────────────────────────────────
    # st.columns(4) creates 4 equal columns for a row of metrics
    col1, col2, col3, col4 = st.columns(4)

    # st.metric() shows a labeled number in a styled card
    # label= is the text above the number, value= is the number to display
    col1.metric("Total Records",     f"{len(df):,}")
    col2.metric("Neighborhoods",     f"{df['Neighborhood'].nunique()}")
    col3.metric("Avg Weekly Waste",  f"{df['Weekly_Waste_Weight_kg'].mean():,.0f} kg")
    col4.metric("Max Weekly Waste",  f"{df['Weekly_Waste_Weight_kg'].max():,.0f} kg")

    st.markdown("---")

    # ── Chart: Monthly Waste Trends ───────────────────────────────────────────
    st.markdown("#### 📅 Monthly Waste Trends")

    # Group by month and calculate average waste for each month
    monthly = df.groupby('Month')['Weekly_Waste_Weight_kg'].mean()

    # Create short month labels for the x-axis (Jan, Feb, ...)
    short_months = [month_names[m][:3] for m in monthly.index]

    # Create the chart using matplotlib
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    # Draw the line chart on axis ax1
    # marker='o' puts a dot at each data point
    ax1.plot(monthly.index, monthly.values, marker='o', color='#2e75b6', linewidth=2.5)

    # fill_between fills the area under the line with a light blue color
    # alpha=0.1 makes it very transparent
    ax1.fill_between(monthly.index, monthly.values, alpha=0.1, color='#2e75b6')

    # Replace numeric tick positions with month name labels
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(short_months)

    ax1.set_ylabel('Average Waste (kg)')
    ax1.set_title('Average Weekly Waste by Month', fontsize=13)
    ax1.grid(True, alpha=0.3)  # faint gridlines

    plt.tight_layout()

    # st.pyplot() renders a matplotlib figure on the Streamlit page
    st.pyplot(fig1)
    plt.close()  # close the figure to free memory

    # ── Chart: Top N Neighborhoods ─────────────────────────────────────────────
    st.markdown("#### 🏘️ Top Neighborhoods by Average Waste")

    # st.slider() lets the user choose how many neighborhoods to show
    n_show = st.slider("How many neighborhoods to display?", 5, 20, 10)

    # Group by neighborhood, calculate mean, sort descending, keep top N
    top_n = df.groupby('Neighborhood')['Weekly_Waste_Weight_kg'].mean()\
              .sort_values(ascending=False).head(n_show)

    # figsize height scales with number of bars so they don't look squished
    fig2, ax2 = plt.subplots(figsize=(10, max(4, n_show * 0.4)))

    # Horizontal bar chart for easier reading of neighborhood names
    bars = ax2.barh(top_n.index, top_n.values, color='#c55a11', edgecolor='white')

    ax2.set_xlabel('Average Weekly Waste (kg)')
    ax2.set_title(f'Top {n_show} Neighborhoods by Average Weekly Waste', fontsize=13)

    # Invert y-axis so the highest bar appears at the top
    ax2.invert_yaxis()

    # Add value labels at the end of each bar
    for bar, val in zip(bars, top_n.values):
        # bar.get_y() + bar.get_height()/2 positions the label vertically centered
        ax2.text(val + 10, bar.get_y() + bar.get_height() / 2,
                 f'{val:,.0f}', va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Neighborhood Deep Dive ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Neighborhood Deep Dive")

    # Dropdown to select a specific neighborhood to inspect
    # key= is a unique identifier so this widget doesn't conflict with others
    selected_n = st.selectbox("Choose a neighborhood:", neighborhoods, key="explorer")

    # Filter the DataFrame to only rows where Neighborhood equals selected_n
    n_data = df[df['Neighborhood'] == selected_n]

    # Show 3 quick metrics for the chosen neighborhood
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Waste", f"{n_data['Weekly_Waste_Weight_kg'].mean():,.0f} kg")
    c2.metric("Minimum Waste", f"{n_data['Weekly_Waste_Weight_kg'].min():,.0f} kg")
    c3.metric("Maximum Waste", f"{n_data['Weekly_Waste_Weight_kg'].max():,.0f} kg")

    # Monthly breakdown bar chart for the selected neighborhood
    n_monthly = n_data.groupby('Month')['Weekly_Waste_Weight_kg'].mean()

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(
        [month_names[m][:3] for m in n_monthly.index],  # short month names on x-axis
        n_monthly.values,                                 # bar heights
        color='#1f4e79', edgecolor='white'
    )
    ax3.set_ylabel('Average Waste (kg)')
    ax3.set_title(f'Monthly Waste Pattern — {selected_n}', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Raw data table inside a collapsible section
    # st.expander() hides content until the user clicks to expand it
    with st.expander("📄 View Raw Data (first 50 rows)"):
        # st.dataframe() renders an interactive scrollable table
        # use_container_width=True makes it fill the available width
        st.dataframe(df.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ How Does This App Work?")

    # st.markdown() renders Markdown text
    # ### = medium heading, ** ** = bold text, - = bullet point
    st.markdown("""
    ### 🤖 What is Machine Learning?
    Machine learning is when we teach a computer to find patterns in data and use those
    patterns to make predictions on new data it has never seen before.

    Instead of writing rules like *"if population > 5000 then waste > 3000 kg"*,
    we give the computer thousands of past examples and let it figure out the rules itself.

    ---

    ### 📊 The Dataset
    We used **10,000 weekly records** from **50 neighborhoods** in Bamenda, Cameroon.
    The data covers multiple years and includes:
    """)

    # st.dataframe() renders a table from a pandas DataFrame
    # We build a small DataFrame to display feature descriptions
    features_df = pd.DataFrame({
        'Feature': [
            'Population Density', 'Number of Businesses', 'Avg Household Size',
            'Market Present', 'Industrial Zone', 'Collection Frequency', 'Month', 'Neighborhood'
        ],
        'Description': [
            'Persons per km² — how crowded the area is',
            'Shops, offices, restaurants, etc.',
            'Average number of people per home',
            'Whether a market exists (1=Yes, 0=No)',
            'Whether the area is industrial (1=Yes, 0=No)',
            'How many times trucks collect waste per week',
            'Month of the year (1=January ... 12=December)',
            'Name of the neighborhood (converted to a number for the model)'
        ],
        'Type': ['Number', 'Number', 'Number', 'Yes/No', 'Yes/No', 'Number', 'Number', 'Category']
    })

    # hide_index=True hides the 0, 1, 2... row numbers on the left
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ---

    ### 🌲 The Model: Random Forest
    We trained 3 models and compared them. Random Forest performed best:

    | Model | R² Score | Meaning |
    |-------|----------|---------|
    | Linear Regression | ~0.50 | Explains 50% of waste variation |
    | **Random Forest** | **~0.91** | **Explains 91% of waste variation ✅** |
    | Gradient Boosting | ~0.88 | Explains 88% of waste variation |

    **Random Forest** works by building 100 different decision trees on random
    subsets of the data. Each tree makes a prediction, and the final answer is
    the average of all 100 trees. This reduces errors caused by any single tree
    being wrong.

    ---

    ### 🚛 How to Use the Predictions

    1. Go to the **Make a Prediction** tab
    2. Fill in the neighborhood details
    3. Click **Predict Waste Generation**
    4. The app tells you the estimated kg and how many trucks to send

    Rule of thumb used in this app:
    - **1 truck = 1,500 kg capacity**
    - If prediction = 3,200 kg → send **2 trucks**

    ---

    ### 👨‍💻 Technologies Used

    | Tool | Purpose |
    |------|---------|
    | Python | Programming language |
    | pandas | Loading and manipulating data |
    | scikit-learn | Building and training ML models |
    | joblib | Saving and loading the trained model |
    | matplotlib | Drawing charts |
    | Streamlit | Building this web app |

    ---
    """)

    # st.info() shows a blue box — good for tips and notes
    st.info("💡 **Beginner tip:** Try changing different values in the Prediction tab and observe how the predicted waste changes. This is a great way to understand which factors have the most impact!")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
# A simple footer at the bottom of every tab
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "</div>",
    unsafe_allow_html=True  # needed to render the HTML div tag
)
