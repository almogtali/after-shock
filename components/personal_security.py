
from datetime import datetime
import json
import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go

def personal_security_purpose():
    """Displays an explanation of the personal security section in the dashboard."""
    st.subheader("Purpose")
    st.write(
        "Analyzing citizens' sense of personal security with regional breakdowns, "
        "providing insights into how different areas experience and perceive threats."
    )

def personal_security_how_to():
    """Displays an explanation of the personal security section in the dashboard."""
    st.subheader("How To Use")
    st.write(
        "Filter period using the slider.\n\n"
        "The district’s colors represent the average percentage of people in the district who feel secure – "
        "the darker the color the more sense of security in this district. "
        "Hover over a district to see the average survey responses for this district in the selected period.\n\n"
        "It is optional to show alarms location, by checking the checkbox. "
        "The dots represent the sum of alarms per locality, "
        "the darker the color the higher number of alarms at this location. "
        "Hover over a dot to see the locality and the sum of alarms at this locality in the selected period."
    )

def rocket_strikes_map(selected_range,show_alarms):
    data_dir = "./data"
    surveyCSV = f"{data_dir}/AggCountiesData.csv"
    alarmsCSV = f"{data_dir}/AggAlarmsData.csv"
    countiesSHP = f"{data_dir}/Mechozot_all/Mechozot_all.shp"

    # Load the data using the new load_data function
    @st.cache_data
    def load_data(surveyCSV, alarmsCSV, countiesSHP):
        gdf = gpd.read_file(countiesSHP)
        data = pd.read_csv(surveyCSV)
        alarms = pd.read_csv(alarmsCSV)

        # Convert date columns to appropriate types
        data["Date"] = pd.to_datetime(data["Date"], errors='coerce').dt.date  # Ensure consistent parsing
        data["Year-Month"] = pd.to_datetime(data["Year-Month"], errors='coerce').dt.to_period('M')
        alarms["Year-Month"] = pd.to_datetime(alarms["Year-Month"], errors='coerce').dt.to_period('M')

        # Convert to English names
        mapping_dict = {
            'מרכז': "Center",
            "תל אביב": "Tel-Aviv",
            "יהודה ושומרון": "Judea and Samaria",
            "ירושלים": "Jerusalem",
            "דרום": "South",
            "צפון": "North",
            "חיפה": "Haifa"
        }
        gdf["machoz"] = gdf["machoz"].replace(mapping_dict)

        return gdf, data, alarms

    # Load data
    gdf, data, alarms = load_data(surveyCSV, alarmsCSV, countiesSHP)


    filtered_data = data[
        (data['Year-Month'] >= selected_range[0]) &
        (data['Year-Month'] <= selected_range[1])
        ]

    if not filtered_data.empty:  # Ensure there is data to avoid errors
        # Filter alarms data based on selected period
        filtered_alarms = alarms[
            (alarms["Year-Month"] >= selected_range[0]) &
            (alarms["Year-Month"] <= selected_range[1] + pd.offsets.MonthEnd())
            ]

        # Group alarms data and calculate sum for period
        grouped_alarms = filtered_alarms.groupby(['data', 'outLat', 'outLong'], as_index=False)['count'].sum().round(2)

        # Group survey data and calculate percentage per answer per county
        avg_by_machoz = (
            filtered_data.groupby("machoz")
            .apply(lambda x: x.iloc[:, 2:].mean().round(2))
            .reset_index()
            .rename(columns={"index": "machoz"})
        )

        # Assuming 'feel_safe' is one of the columns in avg_by_machoz
        # If it's calculated differently, adjust accordingly
        if 'feel_safe' not in avg_by_machoz.columns:
            # Example calculation if 'feel_safe' needs to be computed
            # Adjust the columns as per your actual data
            feel_safe = (
                filtered_data.groupby("machoz").apply(
                    lambda x: (x.iloc[:, 2:4].sum(axis=1).mean() * 100).round(2)
                ).reset_index(name="feel_safe")
            )
            avg_by_machoz = avg_by_machoz.merge(feel_safe, on="machoz", how="left")

        # Merge survey data with GeoDataFrame
        merged_gdf = gdf.merge(avg_by_machoz, on="machoz", how="left").to_crs(epsg=4326)

        # Create Choropleth Map
        choropleth = go.Choroplethmapbox(
            geojson=json.loads(merged_gdf.to_json()),
            locations=merged_gdf["machoz"],
            featureidkey="properties.machoz",
            z=merged_gdf["feel_safe"],
            zmin=merged_gdf["feel_safe"].min(),
            zmax=merged_gdf["feel_safe"].max(),
            colorscale="Blues",
            colorbar=dict(title="Feel Secure (%)"),
            marker=dict(opacity=0.7),
            hovertext=merged_gdf.apply(
                lambda row: (
                    f"<b>Sense of Personal Security in {row['machoz']} District</b><br>"
                    f"Don't know: {row.get('dont know', 'N/A')}%<br>"
                    f"Low: {row.get('low', 'N/A')}%<br>"
                    f"Very Low: {row.get('very low', 'N/A')}%<br>"
                    f"Medium: {row.get('medium', 'N/A')}%<br>"
                    f"High: {row.get('high', 'N/A')}%<br>"
                    f"Very High: {row.get('very high', 'N/A')}%"
                ),
                axis=1
            ),
            hoverinfo="text",
        )

        # Create Scatter Map for Alarms
        scatter = go.Scattermapbox(
            lat=grouped_alarms["outLat"],
            lon=grouped_alarms["outLong"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=5,
                opacity=0.7,
                color=grouped_alarms["count"],
                colorscale=[
                    [0, '#FF002B'],
                    [0.5, '#81001C'],
                    [1, '#000000']
                ],
                cmin=max(grouped_alarms["count"].mean() - grouped_alarms["count"].std() * 2, 0),
                # Minimum value for color range
                cmax=grouped_alarms["count"].mean() + grouped_alarms["count"].std() * 2,
                # Maximum value for color range
                showscale=True,  # Show the color scale legend
                colorbar=dict(
                    title="Alarms Amount",
                    x=1.2,
                ),
            ),
            text=grouped_alarms.apply(lambda row: f"Data: {row['data']}<br>Count: {row['count']}", axis=1),
            hoverinfo="text",
        )

        # Initialize Figure
        fig = go.Figure()
        fig.add_trace(choropleth)

        if show_alarms:
            fig.add_trace(scatter)

        # Update Layout
        fig.update_layout(
            mapbox=dict(
                style="carto-positron",
                center={"lat": 31.45, "lon": 35},
                zoom=6.7,
            ),
            width=900,  # Adjusted width for better visibility
            height=750,  # Adjusted height for better visibility
            # title="Sense of Personal Security",
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No data available for the selected time range. Please select a wider period.")

def show_alerts_statistics():
    # Constants for alerts data
    TOTAL_ALERTS = 35417
    START_DATE = datetime(2023, 10, 7)
    END_DATE = datetime(2024, 11, 30)
    TOTAL_DAYS = (END_DATE - START_DATE).days
    AVG_ALERTS = round(TOTAL_ALERTS / TOTAL_DAYS)
    AVG_FEEL_SECURE = 30.27  # The new metric to display

    # Custom CSS for styling (made slightly smaller)
    st.markdown("""
        <style>
        .big-number {
            font-size: 32px; /* Reduced from 36px */
            font-weight: bold;
            text-align: center;
        }
        .stat-card {
            background-color: transparent;
            border-radius: 10px;
            padding: 8px; /* Reduced padding from 10px */
            text-align: center;
            height: 100%;
        }
        .alert-header {
            text-align: center;
            margin-bottom: 0.3rem; /* Slightly reduced margin */
            font-size: 1rem;       /* Reduced from 1.25rem */
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # Container for the alerts component
    with st.container():

        # Header
        st.subheader("Data Overview")

        # Tabs
        tab1, tab2 = st.tabs(["Counter", "Statistics"])

        with tab1:
            st.markdown(
                f"<p style='text-align: center; font-size: 12px;'>"
                f"From {START_DATE.strftime('%B %d, %Y')} to {END_DATE.strftime('%B %Y')}</p>",
                unsafe_allow_html=True
            )
            st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem;'> <!-- Reduced padding -->
                    <div class='big-number'>{format(TOTAL_ALERTS, ',')}</div>
                    <p style='font-size: 14px;'>Total Alerts Recorded</p> <!-- Slightly smaller font -->
                </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown(
                f"<p style='text-align: center; font-size: 12px;'>"
                f"From {START_DATE.strftime('%B %d, %Y')} to {END_DATE.strftime('%B %Y')}</p>",
                unsafe_allow_html=True
            )
            
            col1, col2, col3 = st.columns(3)  # Three columns to include the new Feel Secure % stat

            with col1:
                st.markdown(f"""
                    <div class='stat-card'>
                        <h3 style='font-size: 14px;'>Daily Average</h3> <!-- Smaller font -->
                        <p class='big-number'>{AVG_ALERTS}</p>
                        <p style='font-size: 12px;'>alerts per day</p> <!-- Smaller font -->
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class='stat-card'>
                        <h3 style='font-size: 14px;'>Total Period</h3> <!-- Smaller font -->
                        <p class='big-number'>{TOTAL_DAYS}</p>
                        <p style='font-size: 12px;'>days</p> <!-- Smaller font -->
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class='stat-card'>
                        <h3 style='font-size: 14px;'>Feel Secure %</h3> <!-- Smaller font -->
                        <p class='big-number'>{AVG_FEEL_SECURE}</p>
                        <p style='font-size: 12px;'>average</p> <!-- Smaller font -->
                    </div>
                """, unsafe_allow_html=True)


