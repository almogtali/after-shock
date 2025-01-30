import pandas as pd
import streamlit as st
from components import (solidarity,
                        overview,
                        personal_security,
                        public_trust)


START_DATE = pd.Timestamp(2023, 10, 1).date()
END_DATE = pd.Timestamp(2024, 11, 30).date()
MONTHS = pd.date_range(start=START_DATE, end=END_DATE, freq='MS').to_period('M')

st.set_page_config(
    page_title="My Dashboard",
    layout="wide"
)

def render_personal_security():
    st.header("Sense of Personal Security")
    col1, col2 = st.columns([1, 2])  # Left for text & controls, right for map

    with col1:
        personal_security.personal_security_text()
        selected_range = st.select_slider(
            key="slider1",
            label="Select Period",
            options=list(MONTHS),
            value=(MONTHS[0], MONTHS[-1])
        )
        show_alarms = st.checkbox(label="Show Alarms", value=True)
        personal_security.show_alerts_statistics()

    with col2:
        personal_security.rocket_strikes_map(selected_range, show_alarms)

def render_public_trust():
    st.header("Public Trust In Institutions And Public Figures")
    row1_left, row1_right = st.columns([1, 2])
    with row1_left:
        public_trust.public_trust_text()
    with row1_right:
        selected_point, data, institutions = public_trust.create_trust_dashboard(
            bibi_data_path="data/bibi.xlsx",
            tzal_data_path="data/tzal.xlsx",
            mishtara_data_path="data/mishtra.xlsx",
            memshala_data_path="data/memshla.xlsx"
        )
    if selected_point:
        st.markdown("---")
        row2_left, row2_right = st.columns([1, 4])
        with row2_left:
            st.write("\n" * 6)  # Add spacing
            selected_demo = st.radio(
                "Choose a demographic dimension:",
                ["All", "District", "Religiousness", "Political stance", "Age"],
                index=0
            )
        with row2_right:
            clicked_institution = selected_point[0]["x"]
            selected_inst_key = next(
                (k for k, val in institutions.items() if val == clicked_institution),
                None
            )
            public_trust.create_demographic_time_series(selected_inst_key, selected_demo, data, institutions)

def render_social_outlook():
    st.title("Israel’s Social Outlook")
    col1, col2 = st.columns([1, 2])

    with col1:
        solidarity.text_for_solidarity()
    with col2:
        solidarity.create_solidarity_dashboard()

if __name__ == "__main__":
    visualization = st.sidebar.radio(
        "Menu",
        [
            "Dashboard Overview",
            "Public Trust In Institutions And Public Figures",
            "Sense of Personal Security",
            "Israel’s Social Outlook"
        ],
        label_visibility="visible"
    )
    if visualization == "Dashboard Overview":
        overview.dashboard_overview()
    elif visualization == "Sense of Personal Security":
        render_personal_security()
    elif visualization == "Public Trust In Institutions And Public Figures":
        render_public_trust()
    elif visualization == "Israel’s Social Outlook":
        render_social_outlook()
