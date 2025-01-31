import pandas as pd
import streamlit as st
import plotly.graph_objects as go

QUESTION_SHORT_FORMS = {
    "Are you or a first-degree family member involved in combat?": "Combat involvement (family)",
    "Are you or a first-degree family member residing near the Gaza envelope or northern border?": "Border residence (family)",
    "How concerned are you about Israel's social situation on the day after the war?": "Post-war concerns",
    "How optimistic are you about Israeli society's ability to recover from the crisis and grow?": "Recovery optimism",
    "Has there been a change in the sense of solidarity in Israeli society at this time?": "Solidarity change"
}

def text_for_solidarity():
    st.markdown("""
    ### Purpose
    Evaluating collective attitudes toward:
    - National sense of solidarity  
    - Social situation the day after the war  
    - Recovery prospects and future potential growth post-crisis  

    ### How To Use
    1. **Choose a subject** you want to display.  
    2. **Choose a visualization type**: aggregated results or results over time.  
    3. **Choose a segmentation** to display the plots by.  
    4. Each subgroup is represented in a different color.  

    - **Aggregated Results**: Shows a bar plot with the average percentage for each answer within a given subgroup.  
    - **Results Over Time**: Displays how the highest score for each subgroup changes throughout the war.
    """)

def create_bar_chart(question_data,full_question,selected_question,subject_label,dimension_label,response_mappings,QUESTION_SHORT_FORMS):

    color_mapping = {
        "Involved in combat (self or first-degree family member)": '#333333',
        "Not involved in combat (self or first-degree family member)": '#E68943',
        "Female": "#8B0000",
        "Male": "#00008B",
        "Living in North/Gaza Envelope": "#00B300",
        "Not Living in North/Gaza Envelope": "#8600E6",
        "Unknown": "#999999"
    }
    
    yes_no_map = {
        "כן": "Involved in combat (self or first-degree family member)",
        "לא": "Not involved in combat (self or first-degree family member)"
    }
    male_fmale = {
        "זכר": "Male",
        "נקבה": "Female",
        "כן": "Living in North/Gaza Envelope",
        "לא": "Not Living in North/Gaza Envelope"
    }

    numeric_cols = [col for col in question_data.columns if col.startswith("a")]
    numeric_cols.sort()

    chart_data = question_data.groupby("sub_subject", as_index=False)[numeric_cols].mean()
    fig = go.Figure()

    # Map columns (a1, a2, ...) to English if available
    if full_question in response_mappings:
        categories = [response_mappings[full_question].get(col, col) for col in numeric_cols]
    else:
        categories = [f"Response {i}" for i in range(1, len(numeric_cols) + 1)]

    solidarity_order = [
        "Solidarity has significantly decreased",
        "Solidarity has somewhat decreased",
        "No change in solidarity",
        "Solidarity has somewhat strengthened",
        "Solidarity has significantly strengthened"
    ]
    concern_order = [
        "Very concerned"
        "Concerned",
        "Moderately concerned",
        "Slightly concerned",
        "Not concerned at all",
    ]
    optimism_order = [
        "Very optimistic"
        "Somewhat optimistic",
        "Neutral",
        "Somewhat pessimistic",
        "Very pessimistic",
    ]

    if "optimism" in selected_question.lower():
        category_order = optimism_order
    elif "concerns" in selected_question.lower():
        category_order = concern_order
    elif "solidarity" in selected_question.lower():
        category_order = solidarity_order
    else:
        category_order = categories

    # Create the bar traces
    for _, row in chart_data.iterrows():
        sub_subject = row["sub_subject"]

        if selected_question == "Are you or a first-degree family member involved in combat?":
            legend_name = yes_no_map.get(sub_subject.strip(), "Unknown")
        else:
            legend_name = male_fmale.get(sub_subject.strip(), "Unknown")

        bar_color = color_mapping.get(legend_name, color_mapping["Unknown"])
        values = [row[col] * 100 for col in numeric_cols]
        text_values = [f"{v:.1f}%" for v in values]

        fig.add_trace(go.Bar(
            name=legend_name,
            x=values,
            y=categories,
            text=text_values,
            textposition="outside",
            textfont=dict(size=14),
            orientation="h",
            marker_color=bar_color,
            hovertemplate="<b>%{y}</b><br>" +
                          f"{legend_name}: " + "%{x:.1f}%" +
                          "<extra></extra>"
        ))

    # SHORTER TITLE
    title_text = f"{subject_label} | {dimension_label} (Aggregated)"

    fig.update_layout(
        title={
            "text": title_text,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24}
        },
        barmode="group",
        height=700,
        width=1200,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5
        },
        font=dict(size=16),
        margin=dict(t=80, b=200, l=100, r=200),
        paper_bgcolor="white",
        plot_bgcolor="rgba(248,249,250,0.5)",
        yaxis={"autorange": "reversed"},
        xaxis=dict(
            range=[-5, 62],
            showgrid=True,
            dtick=10
        )
    )
    if subject_label == "Optimism" or "Concerns" in subject_label:
        fig.update_yaxes(
        categoryorder="array",
        categoryarray=category_order[::-1],
        tickangle=0,
        automargin=True)
    else:
        fig.update_yaxes(
        categoryorder="array",
        categoryarray=category_order,
        tickangle=0,
        automargin=True)
    st.plotly_chart(fig, use_container_width=True, key="bar_chart")


def create_line_plot(
        question_data,
        full_question,
        selected_question,
        subject_label,  # short subject label
        dimension_label,  # short segmentation label
        response_mappings
):
    # Definition of how to sum columns for each question
    aggregations = {
        'האם חל שינוי בתחושת הסולידריות בחברה הישראלית בעת הזו': {
            'name': 'Solidarity has strengthened',
            'columns': ['a1', 'a2']
        },
        'עד כמה אתה אופטימי ביחס ליכולתה של החברה הישראלית להתאושש מהמשבר ולצמוח': {
            'name': 'Optimistic',
            'columns': ['a3', 'a4']
        },
        'עד כמה את.ה מוטרד.ת או לא מוטרד.ת ממצבה החברתי של ישראל ביום שאחרי המלחמה דרג בסולם של 1-5, כאשר 5 = מוטרד מאד ו - 1 = לא מוטרד כלל': {
            'name': 'Concerned',
            'columns': ['a4', 'a5']
        }
    }

    color_mapping = {
        "Involved in combat (self or first-degree family member)": '#333333',
        "Not involved in combat (self or first-degree family member)": '#E68943',
        "Female": "#8B0000",
        "Male": "#00008B",
        "Living in North/Gaza Envelope": "#00B300",
        "Not Living in North/Gaza Envelope": "#8600E6",
        "Unknown": "#999999"
    }

    fig = go.Figure()
    agg_config = aggregations.get(full_question)

    if agg_config:
        aggregated_data = question_data.copy()
        aggregated_data['agg_value'] = aggregated_data[agg_config['columns']].sum(axis=1)

        for sub_subject in aggregated_data['sub_subject'].unique():
            sub_data = aggregated_data[aggregated_data['sub_subject'] == sub_subject].sort_values('date')

            if selected_question == "Are you or a first-degree family member involved in combat?":
                if sub_subject == 'כן':
                    legend_name = "Involved in combat (self or first-degree family member)"
                else:
                    legend_name = "Not involved in combat (self or first-degree family member)"
            else:
                if sub_subject in ('זכר', 'Male'):
                    legend_name = "Male"
                elif sub_subject in ('נקבה', 'Female'):
                    legend_name = "Female"
                elif sub_subject == 'כן':
                    legend_name = "Living in North/Gaza Envelope"
                elif sub_subject == 'לא':
                    legend_name = "Not Living in North/Gaza Envelope"
                else:
                    legend_name = "Unknown"

            color = color_mapping.get(legend_name, color_mapping['Unknown'])

            fig.add_trace(go.Scatter(
                x=sub_data['date'],
                y=sub_data['agg_value'] * 100,
                name=legend_name,
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8),
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    f"{agg_config['name']}: %{{y:.1f}}%<br>"
                    "<extra></extra>"
                )
            ))

        # SHORTER TITLE
        title_text = f"{subject_label} | {dimension_label} (Over Time)"

        all_y_values = [val for trace in fig.data for val in trace.y]
        max_y = min(max(all_y_values) + 5, 100) if all_y_values else 100

        all_dates = [val for trace in fig.data for val in trace.x]
        if all_dates:
            min_date, max_date = min(all_dates), max(all_dates)
        else:
            min_date, max_date = None, None

        fig.update_layout(
            title={
                'text': title_text,
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Date",
            yaxis_title="Percentage (%)",
            height=600,
            showlegend=True,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.3,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'size': 12}
            },
            font=dict(size=14),
            margin=dict(t=100, b=150, l=100, r=100),
            paper_bgcolor='white',
            plot_bgcolor='rgba(248,249,250,0.5)',
            yaxis=dict(
                range=[0, max_y],
                dtick=10,
                gridcolor='lightgray'
            ),
            xaxis=dict(
                gridcolor='lightgray',
                type='date',
                dtick='M1',
                tickformat='%b %Y',
                range=[min_date, max_date] if min_date else None
            )
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        st.plotly_chart(fig, use_container_width=True, key="line_plot")
    else:
        st.error("Question configuration not found")

def create_solidarity_dashboard():

    # 1. Mapping of English question labels to Hebrew question strings
    question_mapping = {
        "Are you or a first-degree family member involved in combat?":
            "האם את.ה או בן משפחה בדרגה ראשונה שלך לוקח חלק בלחימה?",
        "Are you or a first-degree family member residing near the Gaza envelope or northern border?":
            "האם את.ה או בן משפחה בדרגה ראשונה שלך מתגורר בעוטף עזה או בגבול הצפון?",
        "Gender": "מגדר"
    }

    # 2. File selection with predefined questions (English labels)
    file_options = {
        "Has there been a change in the sense of solidarity in Israeli society at this time?": "data/solidarity.xlsx",
        "How concerned are you about Israel's social situation on the day after the war?": "data/matzv_chvrati.xlsx",
        "How optimistic are you about Israeli society's ability to recover from the crisis and grow?": "data/mashber.xlsx"
    }

    # --- NEW: Mapping from long titles → short display names ---
    display_mapping = {
        "Has there been a change in the sense of solidarity in Israeli society at this time?": "Change in national solidarity during wartime",
        "How concerned are you about Israel's social situation on the day after the war?": "Concerns about Israel’s post-war social stability",
        "How optimistic are you about Israeli society's ability to recover from the crisis and grow?": "Optimism about Israel’s ability to recover from the crisis"
    }

    chart_title_mapping = {
        "Has there been a change in the sense of solidarity in Israeli society at this time?": "Solidarity",
        "How concerned are you about Israel's social situation on the day after the war?": "Post-war Concerns",
        "How optimistic are you about Israeli society's ability to recover from the crisis and grow?": "Optimism"
    }
    segmentation_short_mapping = {
        "Involved in combat (self or first-degree family member)": "Involved in combat",
        "Residing near Gaza/northern border (self or first-degree family member)": "Residing near border",
        "Gender": "Gender"
    }

    # -----------------------------------------------------------

    # 3. Response mappings (in Hebrew → displayed as English)
    response_mappings = {
        "עד כמה את.ה מוטרד.ת או לא מוטרד.ת ממצבה החברתי של ישראל ביום שאחרי המלחמה דרג בסולם של 1-5, כאשר 5 = מוטרד מאד ו - 1 = לא מוטרד כלל": {
            "a1": "Not concerned at all",
            "a2": "Slightly concerned",
            "a3": "Moderately concerned",
            "a4": "Concerned",
            "a5": "Very concerned"
        },
        "עד כמה אתה אופטימי ביחס ליכולתה של החברה הישראלית להתאושש מהמשבר ולצמוח": {
            "a4": "Very optimistic",
            "a3": "Somewhat optimistic",
            "a2": "Somewhat pessimistic",
            "a1": "Very pessimistic"
        },
        "האם חל שינוי בתחושת הסולידריות בחברה הישראלית בעת הזו": {
            "a1": "Solidarity has strengthened significantly",
            "a2": "Solidarity has somewhat strengthened",
            "a3": "No change in solidarity",
            "a4": "Solidarity has somewhat decreased",
            "a5": "Solidarity has significantly decreased"
        }
    }

    # 4. The original 'predefined_questions' list (English labels)
    predefined_questions = list(question_mapping.keys())

    # 5. Concise statements to replace the question text (for sub-subject segmentation)
    concise_statements = {
        "Are you or a first-degree family member involved in combat?": "Involved in combat (self or first-degree family member)",
        "Are you or a first-degree family member residing near the Gaza envelope or northern border?": "Residing near Gaza/northern border (self or first-degree family member)",
        "Gender": "Gender"
    }
    statement_to_question = {v: k for k, v in concise_statements.items()}

    # --- REPLACE the standard selectbox with short display names ---
    # Generate a display list of short names for the dropdown
    display_list = [display_mapping[orig_key] for orig_key in file_options.keys()]
    # Create a selectbox using the short names
    selected_display = st.selectbox("Select Subject to Display:", display_list)
    # Convert the user's short-name choice back to the original key
    selected_file_original = {v: k for k, v in display_mapping.items()}[selected_display]
    subject_label_for_chart = chart_title_mapping[selected_file_original]
    # ---------------------------------------------------------------

    col1, col2 = st.columns([0.8, 2.2])

    with col1:
        viz_type = st.radio(
            "Select Visualization Type:",
            ["Aggregated results", "Results over time"]
        )

    with col2:
        selected_statement = st.selectbox(
            "Select segmentation to display:",
            [concise_statements[q] for q in predefined_questions]
        )
    short_seg_label = segmentation_short_mapping.get(selected_statement, selected_statement)

    # Convert the selected statement back to the original question key
    selected_question = statement_to_question.get(selected_statement, selected_statement)

    # 7. Find the Hebrew question string
    hebrew_question = question_mapping.get(selected_question)

    # 8. Load the data (use selected_file_original in place of 'selected_file')
    try:
        df = pd.read_excel(file_options[selected_file_original])
        df["date"] = pd.to_datetime(df["date"])
    except FileNotFoundError:
        st.error(f"The selected survey file '{selected_file_original}' was not found. Please check the file path.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return

    # 9. Filter data for this Hebrew question
    if hebrew_question:
        question_data = df[df["subject"] == hebrew_question].copy()
        question_data = question_data[question_data["sub_subject"] != "Total"]
    else:
        st.error("Selected question mapping not found.")
        return

    # Attempt to retrieve the full question text if present
    full_question = df["q_full"].iloc[0] if not df.empty else None

    # 10. Choose which visualization to draw
    if viz_type == "Aggregated results":
        # create_bar_chart(question_data, full_question, selected_question, response_mappings, QUESTION_SHORT_FORMS)
        create_bar_chart(
            question_data,
            full_question,
            selected_question,
            subject_label=subject_label_for_chart,  # or whatever short label you want
            dimension_label=short_seg_label,  # or whichever segmentation label
            response_mappings=response_mappings,
            QUESTION_SHORT_FORMS=QUESTION_SHORT_FORMS
        )
    else:

        create_line_plot(
            question_data,
            full_question,
            selected_question,
            subject_label=subject_label_for_chart,  # short subject label
            dimension_label=short_seg_label,  # short segmentation label
            response_mappings=response_mappings
        )

