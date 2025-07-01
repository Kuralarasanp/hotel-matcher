import streamlit as st
import pandas as pd
import numpy as np

st.title("🏨Hotel Comparable Matching with Filters & Multi-Select")

# ======= 1. Input Data Loading (Manual file upload for simplicity) =======
uploaded_file = st.file_uploader("Upload your Excel file (WA Hotel Comparable Sheet)", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    # Prepare columns
    df['No. of Rooms'] = pd.to_numeric(df['No. of Rooms'], errors='coerce')
    df['Market Value-2024'] = pd.to_numeric(df['Market Value-2024'], errors='coerce')
    df['2024 VPR'] = pd.to_numeric(df['2024 VPR'], errors='coerce')
    df = df.dropna(subset=['No. of Rooms', 'Market Value-2024', '2024 VPR'])

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
    df['Hotel Class Order'] = df['Hotel Class'].map(hotel_class_map)
    df = df.dropna(subset=['Hotel Class Order'])
    df['Hotel Class Order'] = df['Hotel Class Order'].astype(int)

    # ======= 2. Multi-select Hotel Names =======
    hotel_names = df['Project / Hotel Name'].unique().tolist()
    select_all = st.checkbox("🏨 Select All Hotels")
    selected_hotels = st.multiselect(
        "Select Project / Hotel Name(s)",
        options=hotel_names,
        default=hotel_names if select_all else []
    )

    # ======= 3. Filters =======
    st.write("### Filters")

    mv_min_pct = st.slider("🔽 Market Value Min Filter %", 0, 100, 80)
    mv_max_pct = st.slider("🔼 Market Value Max Filter %", 100, 200, 120)

    vpr_min_pct = st.slider("🔽 VPR Min Filter %", 0, 100, 80)
    vpr_max_pct = st.slider("🔼 VPR Max Filter %", 100, 200, 120)

    # Validate filter ranges
    if mv_min_pct > mv_max_pct:
        st.error("Market Value Min % should not be greater than Max %")
    if vpr_min_pct > vpr_max_pct:
        st.error("VPR Min % should not be greater than Max %")

    # ======= 4. Prepare base & result columns =======
    base_columns = [
        'Project / Hotel Name',
        'State',
        'Property County',
        'No. of Rooms',
        'Market Value-2024',
        '2024 VPR',
        'Hotel Class',
        'Hotel Class Order'
    ]
    result_columns = [col for col in df.columns if col not in base_columns]

    max_results_per_row = 5

    # ======= 5. Run Matching Button =======
    if st.button("🚀 Run Matching"):

        if not selected_hotels:
            st.warning("Please select at least one Project / Hotel Name to run matching.")
        elif mv_min_pct > mv_max_pct or vpr_min_pct > vpr_max_pct:
            st.warning("Fix filter percentage ranges before running matching.")
        else:
            # Filter base dataframe by selected hotels (no duplicate removal)
            base_df = df[df['Project / Hotel Name'].isin(selected_hotels)].reset_index(drop=True)

            output_rows = []

            def get_least_one(df_sub):
                return df_sub.sort_values(['Market Value-2024', '2024 VPR'], ascending=[True, True]).head(1)

            def get_top_one(df_sub):
                return df_sub.sort_values(['Market Value-2024', '2024 VPR'], ascending=[False, False]).head(1)

            def get_nearest_three(df_sub, target_mv, target_vpr):
                df_sub = df_sub.copy()
                df_sub['distance'] = ((df_sub['Market Value-2024'] - target_mv) ** 2 + (df_sub['2024 VPR'] - target_vpr) ** 2) ** 0.5
                return df_sub.sort_values('distance').head(3).drop(columns='distance')

            for idx in range(len(base_df)):
                base_row = base_df.iloc[idx]
                base_mv = base_row['Market Value-2024']
                base_vpr = base_row['2024 VPR']
                base_order = base_row['Hotel Class Order']

                # Allowed hotel class orders logic
                if base_order == 1:
                    allowed_orders = [1, 2, 3]
                elif base_order == 2:
                    allowed_orders = [1, 2, 3, 4]
                elif base_order == 3:
                    allowed_orders = [2, 3, 4, 5]
                elif base_order == 4:
                    allowed_orders = [3, 4, 5, 6]
                elif base_order == 5:
                    allowed_orders = [4, 5, 6, 7]
                elif base_order == 6:
                    allowed_orders = [5, 6, 7, 8]
                elif base_order == 7:
                    allowed_orders = [6, 7, 8]
                elif base_order == 8:
                    allowed_orders = [7, 8]
                else:
                    allowed_orders = []

                subset = df[df.index != base_row.name]

                # Apply filters
                mask = (
                    (subset['Project / Hotel Name'] != base_row['Project / Hotel Name']) &
                    (subset['State'] == base_row['State']) &
                    (subset['Property County'] == base_row['Property County']) &
                    (subset['No. of Rooms'] < base_row['No. of Rooms']) &
                    (subset['Market Value-2024'].between(base_mv * mv_min_pct/100, base_mv * mv_max_pct/100)) &
                    (subset['2024 VPR'].between(base_vpr * vpr_min_pct/100, base_vpr * vpr_max_pct/100)) &
                    (subset['Hotel Class Order'].isin(allowed_orders))
                )

                matching_rows = subset[mask] \
                    .drop_duplicates(subset=['Project / Hotel Name', 'Owner Street Address', 'Owner Name/ LLC Name'], keep='first')

                out_dict = {}
                for col in base_columns:
                    out_dict[col] = base_row.get(col, np.nan)

                if not matching_rows.empty:
                    nearest_3 = get_nearest_three(matching_rows, base_mv, base_vpr)
                    remaining_after_nearest = matching_rows[~matching_rows.index.isin(nearest_3.index)]
                    least_1 = get_least_one(remaining_after_nearest)
                    remaining_after_least = remaining_after_nearest[~remaining_after_nearest.index.isin(least_1.index)]
                    top_1 = get_top_one(remaining_after_least)

                    selected_rows = pd.concat([nearest_3, least_1, top_1]).reset_index(drop=True)
                    selected_rows = selected_rows.head(max_results_per_row)
                    result_count = len(selected_rows)

                    out_dict['Matching Results Count / Status'] = f"Total: {len(matching_rows)} | Selected: {result_count}"

                    for i in range(max_results_per_row):
                        prefix = f"RESULT {i + 1}-"
                        if i < result_count:
                            row = selected_rows.iloc[i]
                            for col in result_columns:
                                out_dict[prefix + col] = row.get(col, '')
                        else:
                            for col in result_columns:
                                out_dict[prefix + col] = ''
                else:
                    out_dict['Matching Results Count / Status'] = "No_Match_Case"
                    for i in range(max_results_per_row):
                        prefix = f"RESULT {i + 1}-"
                        for col in result_columns:
                            out_dict[prefix + col] = ''

                output_rows.append(out_dict)

            # Final output dataframe
            output_df = pd.DataFrame(output_rows)

            # ======= 6. Overall stats =======
            total_rows = len(base_df)
            matching_len = output_df[output_df['Matching Results Count / Status'] != "No_Match_Case"].shape[0]
            no_matching_len = total_rows - matching_len
            result_len = output_df[[col for col in output_df.columns if col.startswith('RESULT 1-')]].dropna(how='all').shape[0]

            st.markdown(f"**Overall Data Length:** {total_rows}")
            st.markdown(f"**Matching Rows:** {matching_len}")
            st.markdown(f"**No Matching Rows:** {no_matching_len}")
            st.markdown(f"**Rows with Results:** {result_len}")

            # ======= 7. Show final results & download option =======
            st.write("✅ Final Matching Results")
            st.dataframe(output_df)

            def to_excel(df_to_save):
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_to_save.to_excel(writer, index=False, sheet_name='Matching Results')
                    writer.save()
                processed_data = output.getvalue()
                return processed_data

            excel_data = to_excel(output_df)
            st.download_button(
                label="Download Final Results as Excel",
                data=excel_data,
                file_name='hotel_matching_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

else:
    st.info("Upload your Excel file to begin.")
