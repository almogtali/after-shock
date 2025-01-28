
import streamlit as st

def dashboard_overview():
    
    st.title("War of Iron Swords: Public Opinion Analytics Dashboard")

    # Overview Section
    st.header("Overview")
    st.write("""
    This dashboard visualizes monthly public opinion surveys related to the War of Iron Swords 
    to provide insights into the ongoing public sentiment along the conflict period.
    """)

    # Data Sources Section
    st.header("Data Sources")

    # Public Opinion Research
    st.subheader("Public Opinion Research")
    st.write("""
    - **Monthly surveys** conducted among the Israeli population by the **INSS** 
      ([Institute for National Security Studies](#)).
    - **Sample size**: ~580 respondents per survey.
    - **Demographics**: Jewish Israeli adults (18+).
    - **Sampling methodology**: Representative of the national adult population.
    - **Statistical validity**: ±3.5% margin of error at **95% confidence level**.
    """)

    # Alarms Data
    st.subheader("Alarms Data")
    st.write("""
    - **Real-time recordings** of alerts regarding **rocket launches** and **hostile aircraft intrusions**.
    - Data collected by scraping from **Pikud HaOref** ([Home Front Command](#)).
    """)

    # Purpose Section
    st.header("Purpose")
    st.write("""
    This dashboard serves as a tool for:
    - **Tracking** evolving public sentiment throughout the war.
    - **Analyzing** relations between security events and public opinion.
    - **Providing** **access** to war-related insights.
    """)

    # Focus Areas
    st.header("Focus Areas")

    # 1. Public Trust
    st.subheader("1. Public Trust In Institutions And Public Figures")
    st.write("Measuring confidence levels in public institutions throughout the war period.")

    # 2. Personal Security
    st.subheader("2. Sense of Personal Security")
    st.write("""
    Analyzing citizens' sense of personal security with regional breakdowns, 
    providing insights into how different areas experience and perceive threats.
    """)

    # 3. National Perspective
    st.subheader("3. Israel’s Social Outlook")
    st.write("""
    Evaluating collective attitudes toward:
    - **National unity and solidarity**.
    - **Current social conditions**.
    - **Recovery prospects and future growth potential post-crisis**.
    """)
