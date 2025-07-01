import streamlit as st
import pandas as pd
import numpy as np
import io

# Mapping for hotel class
hotel_class_map = {
    "Budget (Low End)": 1,
    "Economy (Name Brand)": 2,
    "Midscale": 3,
    "Upper Midscale": 4,
    "Upscale": 5,
    "Upper Upscale First Class": 6,
    "Luxury Class": 7,
    "Independent Hotel": 8
}

# Match functions (you can keep or modify as needed)
def get_least_one(df):
    return df.sort_values(['Market Value-2024', '2024 VPR'], ascending=[True, True]).head(1)

def get_top_one(df):
    return df.sort_values(['Market Value-2024', '2024 VPR'], ascending=[False, False]).head(1)

def get_nearest_three(df, target_value_mv, target_value_vpr):
    df = df.copy()
    df['distance'] = ((df['Market Value-2024'] - target_value_mv) ** 2 + (df['2024 VPR'] - target_value_vpr) ** 2) ** 0.5
    return df.sort_values('distance').head(3).drop(columns='distance')

# Streamlit UI
st.title("🏨 Hotel Comparable Matcher Tool")

uploaded_file = st.file_uploader("📤 Upload Excel File", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    # Clean and preprocess
    cols_to_numeric = ['No. of Rooms', 'Market Value-2024', '2024 VPR']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols_to_numeric)

    df['Hotel Class Order'] = df['Hotel Class'].map(hotel_class_map)
    df = df.dropna(subset=['Hotel Class Order'])
    df['Hotel Class Order'] = df['Hotel Class Order'].astype(int)

    hotel_names = sorted(df['Project / Hotel Name'].dropna().unique())
    selected_hotels = st.multiselect(
        "🏨 Select Project / Hotel Name(s)",
        options=["[SELECT ALL]"] + list(hotel_names),
        default=["[SELECT ALL]"]
    )
    if "[SELECT ALL]" in selected_hotels:
        selected_hotels = hotel_names

    # Market Value Min and Max % inputs
    col1, col2 = st.columns(2)
    with col1:
        mv_min_pct = st.number_input(
            "🔽 Market Value Min Filter %", min_value=0.0, max_value=500.0, value=80.0, step=1.0
        )
    with col2:
        mv_max_pct = st.number_input(
            "🔼 Market Value Max Filter %", min_value=mv_min_pct, max_value=500.0, value=120.0, step=1.0
        )

    # VPR Min and Max % inputs
    col3, col4 = st.columns(2)
    with col3:
        vpr_min_pct = st.number_input(
            "🔽 VPR Min Filter %", min_value=0.0, max_value=500.0, value=80.0, step=1.0
        )
    with col4:
        vpr_max_pct = st.number_input(
            "🔼 VPR Max Filter %", min_value=vpr_min_pct, max_value=500.0, value=120.0, step=1.0
        )

    if st.button("🚀 Run Matching"):
        results_rows = []

        total_match_case_rows = 0
        total_base_hotels_with_matches = 0

        all_columns = [col for col in df.columns]

        for hotel_name in selected_hotels:
            try:
                base_row = df[df['Project / Hotel Name'] == hotel_name].iloc[0]
                base_market_val = base_row['Market Value-2024']
                base_vpr = base_row['2024 VPR']
                base_order = base_row['Hotel Class Order']

                allowed_orders = {
                    1: [1, 2, 3],
                    2: [1, 2, 3, 4],
                    3: [2, 3, 4, 5],
                    4: [3, 4, 5, 6],
                    5: [4, 5, 6, 7],
                    6: [5, 6, 7, 8],
                    7: [6, 7, 8],
                    8: [7, 8]
                }.get(base_order, [])

                # Exclude only the exact same row by index
                subset = df[df.index != base_row.name]

                mask = (
                    (subset['State'] == base_row['State']) &
                    (subset['Property County'] == base_row['Property County']) &
                    (subset['No. of Rooms'] < base_row['No. of Rooms']) &
                    (subset['Market Value-2024'].between(base_market_val * (mv_min_pct / 100), base_market_val * (mv_max_pct / 100))) &
                    (subset['2024 VPR'].between(base_vpr * (vpr_min_pct / 100), base_vpr * (vpr_max_pct / 100))) &
                    (subset['Hotel Class Order'].isin(allowed_orders))
                )

                filtered_subset = subset[mask]

                total_match_case_rows += len(filtered_subset)

                if not filtered_subset.empty:
                    total_base_hotels_with_matches += 1

                    # Output one row per matched comparable with base hotel info
                    for _, match_row in filtered_subset.iterrows():
                        combined_row = {}

                        # Base hotel info prefix: "Base - "
                        for col in all_columns:
                            combined_row[f"Base - {col}"] = base_row[col]

                        # Matched comparable info prefix: "Match - "
                        for col in all_columns:
                            combined_row[f"Match - {col}"] = match_row[col]

                        combined_row['Matching Results Count / Status'] = f"Total Matches for base hotel: {len(filtered_subset)}"
                        results_rows.append(combined_row)
                else:
                    # If no match found, add a row for base hotel with no matches
                    combined_row = {}
                    for col in all_columns:
                        combined_row[f"Base - {col}"] = base_row[col]
                    for col in all_columns:
                        combined_row[f"Match - {col}"] = None
                    combined_row['Matching Results Count / Status'] = 'No_Match_Case'
                    results_rows.append(combined_row)

            except Exception as e:
                st.error(f"❌ Error processing hotel '{hotel_name}': {e}")

        # Show summary
        st.markdown(f"**Total rows in file:** {len(df)}")
        st.markdown(f"**Match Cases with results:** {total_base_hotels_with_matches}")
        st.markdown(f"**Total matching result rows:** {total_match_case_rows}")

        if results_rows:
            result_df = pd.DataFrame(results_rows)
            st.success("✅ Matching Completed")
            st.dataframe(result_df)

            output = io.BytesIO()
            result_df.to_excel(output, index=False)
            st.download_button(
                label="📥 Download Result as Excel",
                data=output.getvalue(),
                file_name="hotel_matching_result_all_matches.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
