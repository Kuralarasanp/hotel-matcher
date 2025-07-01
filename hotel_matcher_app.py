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

# Match functions
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

    # Combine columns into a single string for uniqueness
    combined = df[['Project / Hotel Name', 'Owner Name/ LLC Name', 'Owner Street Address']].dropna()
    combined_str = combined.apply(lambda row: f"{row['Project / Hotel Name']} | {row['Owner Name/ LLC Name']} | {row['Owner Street Address']}", axis=1)

    hotel_names = sorted(combined_str.unique())

    selected_hotels = st.multiselect(
        "🏨 Select Project / Hotel Name(s)",
        options=["[SELECT ALL]"] + list(hotel_names),
        default=["[SELECT ALL]"]
    )

    if "[SELECT ALL]" in selected_hotels:
        selected_hotels = hotel_names

    # Separate min and max input fields for Market Value Filter %
    col1, col2 = st.columns(2)
    with col1:
        mv_min = st.number_input("🔽 Market Value Min Filter %", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
    with col2:
        mv_max = st.number_input("🔼 Market Value Max Filter %", min_value=mv_min, max_value=500.0, value=120.0, step=1.0)

    # Separate min and max input fields for VPR Filter %
    col3, col4 = st.columns(2)
    with col3:
        vpr_min = st.number_input("🔽 VPR Min Filter %", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
    with col4:
        vpr_max = st.number_input("🔼 VPR Max Filter %", min_value=vpr_min, max_value=500.0, value=120.0, step=1.0)

    match_columns = [
        'Project / Hotel Name', 'State', 'Property County',
        'No. of Rooms', 'Market Value-2024', '2024 VPR',
        'Hotel Class', 'Hotel Class Order'
    ]
    all_columns = [col for col in df.columns if col != 'Hotel Class Order'] + ['Hotel Class Order']
    max_results_per_row = 5

    if st.button("🚀 Run Matching"):
        results_rows = []

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

                subset = df[df['Project / Hotel Name'] != hotel_name]

                mask = (
                    (subset['State'] == base_row['State']) &
                    (subset['Property County'] == base_row['Property County']) &
                    (subset['No. of Rooms'] < base_row['No. of Rooms']) &
                    (subset['Market Value-2024'].between(base_market_val * (mv_min / 100), base_market_val * (mv_max / 100))) &
                    (subset['2024 VPR'].between(base_vpr * (vpr_min / 100), base_vpr * (vpr_max / 100))) &
                    (subset['Hotel Class Order'].isin(allowed_orders))
                )

                matching_rows = subset[mask].drop_duplicates(
                    subset=['Project / Hotel Name', 'Owner Street Address', 'Owner Name/ LLC Name'], keep='first'
                )

                base_data = base_row[match_columns].to_dict()

                if not matching_rows.empty:
                    nearest_3 = get_nearest_three(matching_rows, base_market_val, base_vpr)
                    remaining_after_nearest = matching_rows[~matching_rows.index.isin(nearest_3.index)]
                    least_1 = get_least_one(remaining_after_nearest)
                    remaining_after_least = remaining_after_nearest[~remaining_after_nearest.index.isin(least_1.index)]
                    top_1 = get_top_one(remaining_after_least)

                    selected_rows = pd.concat([nearest_3, least_1, top_1]).reset_index(drop=True)
                    result_count = len(selected_rows)

                    combined_row = base_data.copy()
                    combined_row['Matching Results Count / Status'] = f"Total: {len(matching_rows)} | Selected: {result_count}"

                    for idx in range(max_results_per_row):
                        prefix = f"Result {idx + 1} - "
                        if idx < result_count:
                            match_row = selected_rows.iloc[idx]
                            for col in all_columns:
                                combined_row[prefix + col] = match_row[col]
                        else:
                            for col in all_columns:
                                combined_row[prefix + col] = None

                    results_rows.append(combined_row)
                else:
                    combined_row = base_data.copy()
                    combined_row['Matching Results Count / Status'] = 'No_Match_Case'
                    for idx in range(max_results_per_row):
                        prefix = f"Result {idx + 1} - "
                        for col in all_columns:
                            combined_row[prefix + col] = None
                    results_rows.append(combined_row)

            except Exception as e:
                st.error(f"❌ Error processing hotel '{hotel_name}': {e}")

        if results_rows:
            result_df = pd.DataFrame(results_rows)
            st.success("✅ Matching Completed")
            st.dataframe(result_df)

            output = io.BytesIO()
            result_df.to_excel(output, index=False)
            st.download_button(
                label="📥 Download Result as Excel",
                data=output.getvalue(),
                file_name="hotel_matching_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
