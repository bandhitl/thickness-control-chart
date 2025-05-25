import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# SPC calculation function
@st.cache_data
def calculate_spc(data):
    # Calculate X-bar (mean of thicknesses for each row)
    data['X̄'] = data[['Thickness1', 'Thickness2', 'Thickness3']].mean(axis=1)
    # Calculate R (range of thicknesses for each row)
    data['R'] = data[['Thickness1', 'Thickness2', 'Thickness3']].max(axis=1) - data[['Thickness1', 'Thickness2', 'Thickness3']].min(axis=1)
    return data

# Cp & Cpk calculation
def calculate_cp_cpk(data, usl, lsl):
    # Calculate standard deviation of X-bar values
    std_dev = data['X̄'].std()
    # Calculate mean of X-bar values
    mean = data['X̄'].mean()

    # Calculate Cp
    cp = (usl - lsl) / (6 * std_dev) if std_dev > 0 else np.nan
    # Calculate Cpk
    cpk_upper = (usl - mean) / (3 * std_dev) if std_dev > 0 else np.nan
    cpk_lower = (mean - lsl) / (3 * std_dev) if std_dev > 0 else np.nan
    cpk = min(cpk_upper, cpk_lower) if std_dev > 0 else np.nan
    return cp, cpk

# Western/Nelson rules (simplified set)
def apply_special_rules(data, center_line_target):
    xbar_series = data['X̄']
    # Standard deviation of the X-bar series (process variation)
    process_std_dev = xbar_series.std()
    rule_flags_list = []

    # Helper function to check for trends (6 points increasing or decreasing)
    def trend_check(series_segment):
        if len(series_segment) < 6:
            return False
        is_increasing = all(series_segment.iloc[i] < series_segment.iloc[i+1] for i in range(len(series_segment)-1))
        is_decreasing = all(series_segment.iloc[i] > series_segment.iloc[i+1] for i in range(len(series_segment)-1))
        return is_increasing or is_decreasing

    for i in range(len(data)):
        current_flags = []

        # Rule 1: Point outside 3σ from the center_line_target (using process_std_dev)
        # This rule checks if a point is too far from the target, relative to process variation.
        if process_std_dev > 0 and abs(xbar_series.iloc[i] - center_line_target) > 3 * process_std_dev:
            current_flags.append("Rule 1: >3σ from Target")

        # Rule 2: 6 points trending up or down
        if i >= 5: # Need at least 6 points (0 to 5 makes 6 points)
            segment_for_trend = xbar_series.iloc[i-5:i+1]
            if trend_check(segment_for_trend):
                current_flags.append("Rule 2: 6 point trend")

        # Rule 3: 7 points on the same side of the center_line_target
        if i >= 6: # Need at least 7 points (0 to 6 makes 7 points)
            segment_for_side = xbar_series.iloc[i-6:i+1]
            if all(x > center_line_target for x in segment_for_side) or \
               all(x < center_line_target for x in segment_for_side):
                current_flags.append("Rule 3: 7 points same side")

        rule_flags_list.append(", ".join(current_flags) if current_flags else "")

    data['Rule Violations'] = rule_flags_list
    return data

# Streamlit UI
st.set_page_config(layout="wide") # Use wide layout for better chart display
st.title("⚙️ Thickness Control Chart & Process Capability Analysis")
st.markdown("Enter 3 thickness values for each time point. USL/LSL are used for Cp/Cpk and visual guides.")

# Inputs for Product and Machine
with st.sidebar:
    st.header("Process Information")
    product_name = st.text_input("Product Name:", value="Sample Product")
    machine_name = st.text_input("Machine Name:", value="Machine Alpha")

    st.header("Data Entry Setup")
    num_rows = st.number_input("Number of time points (subgroups):", min_value=1, max_value=100, value=10, step=1)

    st.header("Specification Limits")
    usl = st.number_input("Upper Specification Limit (USL):", value=2.60, format="%.3f")
    lsl = st.number_input("Lower Specification Limit (LSL):", value=2.40, format="%.3f")
    
    # Target Center Line (midpoint of USL/LSL)
    # Note: For SPC, CL on X-bar chart is often grand average of X-bars. Here, it's target.
    cl_target = (usl + lsl) / 2
    st.write(f"Target Center Line (CL): {cl_target:.3f}")


# Form for data input
st.subheader("Thickness Data Input")
with st.form(key="thickness_form"):
    time_inputs_list = []
    thickness_inputs_list = []

    # Create a header row for the input table
    cols_header = st.columns([2, 1, 1, 1])
    cols_header[0].markdown("**Time**")
    cols_header[1].markdown("**T1**")
    cols_header[2].markdown("**T2**")
    cols_header[3].markdown("**T3**")

    for i in range(num_rows):
        # Use columns for layout within the loop
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            # Display the "Time X" identifier for the row
            st.markdown(f"Time {i+1}") 

            # Time input field
            default_hour = 9 + i
            default_minute = "00"
            if default_hour >= 24: # Basic wrap-around for example
                default_hour -= 24
            time_val = st.text_input(
                label=f"Input field for Time {i+1}", # Descriptive label for accessibility, hidden from view
                key=f"time_{i}",
                value=f"{default_hour:02d}:{default_minute}",
                label_visibility="collapsed", # This makes the input box align with T1, T2, T3 inputs
                placeholder="HH:MM"
            )
        with col2:
            # Example default values for thickness, centered around (LSL+USL)/2
            t1 = st.number_input(f"T1_{i+1}", key=f"t1_{i}", value=round(np.random.normal(cl_target, 0.05), 3), format="%.3f", label_visibility="collapsed")
        with col3:
            t2 = st.number_input(f"T2_{i+1}", key=f"t2_{i}", value=round(np.random.normal(cl_target, 0.05), 3), format="%.3f", label_visibility="collapsed")
        with col4:
            t3 = st.number_input(f"T3_{i+1}", key=f"t3_{i}", value=round(np.random.normal(cl_target, 0.05), 3), format="%.3f", label_visibility="collapsed")

        time_inputs_list.append(time_val)
        thickness_inputs_list.append((t1, t2, t3))

    submit_button = st.form_submit_button(label='🎯 Generate Control Chart & Analysis')

if submit_button:
    # Basic validation: Check if any thickness input is zero (common uninitialized value for number_input)
    if any(t == 0.0 for row_thicknesses in thickness_inputs_list for t in row_thicknesses):
        st.warning("⚠️ Some thickness values are 0.0. Please ensure all data points are entered correctly.")

    # Create DataFrame from inputs
    df = pd.DataFrame({
        'Time': time_inputs_list,
        'Thickness1': [x[0] for x in thickness_inputs_list],
        'Thickness2': [x[1] for x in thickness_inputs_list],
        'Thickness3': [x[2] for x in thickness_inputs_list]
    })

    st.markdown("---")
    st.subheader("📊 Results & Analysis")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**Product:** {product_name}")
    with col_info2:
        st.markdown(f"**Machine:** {machine_name}")


    # Calculate SPC metrics (X-bar, R)
    # Pass a copy to avoid modifying the original df if calculate_spc modifies in-place (though @st.cache_data handles input immutability)
    result_df = calculate_spc(df.copy())
    
    # Calculate Cp, Cpk
    cp_val, cpk_val = calculate_cp_cpk(result_df, usl, lsl)
    
    # Check for out-of-specification points
    result_df['Out of Spec'] = (result_df['X̄'] > usl) | (result_df['X̄'] < lsl)
    
    # Apply special rules (using the target CL for rule checking)
    result_df = apply_special_rules(result_df, cl_target)

    # Display Process Capability
    st.markdown("### 📈 Process Capability")
    col_cp, col_cpk = st.columns(2)
    with col_cp:
        st.metric(label="Cp (Process Potential)", value=f"{cp_val:.3f}" if not np.isnan(cp_val) else "N/A")
    with col_cpk:
        st.metric(label="Cpk (Process Performance)", value=f"{cpk_val:.3f}" if not np.isnan(cpk_val) else "N/A")

    if np.isnan(cp_val) or np.isnan(cpk_val):
        st.info("Cp/Cpk cannot be calculated (e.g., standard deviation is zero or insufficient data).")
    elif cp_val < 1.00 or cpk_val < 1.00:
        st.error("🚨 Cp or Cpk is below 1.00. The process is generally considered not capable.")
    elif cp_val < 1.33 or cpk_val < 1.33:
        st.warning("⚠️ Cp or Cpk is below 1.33. The process capability may be marginal; improvement is recommended.")
    else:
        st.success("✅ Cp and Cpk are 1.33 or above. The process is generally considered capable.")

    # Warnings for out-of-spec or rule violations
    if result_df['Out of Spec'].any():
        st.error("🔴 Some data points (X̄) are outside the specification limits (USL/LSL). Review data and X̄ Chart.")
    if result_df['Rule Violations'].str.len().sum() > 0: # Check if any rule violation string is non-empty
        st.warning("📌 Special rules triggered in the X̄ data (e.g., trend, points on same side). See X̄ Chart and data table for details.")

    # Display Charts in columns
    st.markdown("### 📉 Control Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### X̄ Chart (Average Thickness)")
        fig_xbar, ax_xbar = plt.subplots(figsize=(10, 5)) # Consistent naming
        ax_xbar.plot(result_df['Time'], result_df['X̄'], marker='o', linestyle='-', color='dodgerblue', label='X̄ (Subgroup Average)')
        
        # Plot Target CL, USL, LSL
        ax_xbar.axhline(cl_target, color='green', linestyle='--', label=f'Target CL ({cl_target:.3f})')
        ax_xbar.axhline(usl, color='red', linestyle=':', label=f'USL ({usl:.3f})')
        ax_xbar.axhline(lsl, color='red', linestyle=':', label=f'LSL ({lsl:.3f})')

        # Highlight points violating rules or out of spec
        for i, row in result_df.iterrows():
            if row['Out of Spec'] or row['Rule Violations']:
                ax_xbar.plot(row['Time'], row['X̄'], marker='X', markersize=10, color='orangered', linestyle='none')
                if row['Rule Violations']: # Add annotation for rule violations
                     ax_xbar.annotate(row['Rule Violations'], (row['Time'], row['X̄']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')


        ax_xbar.set_title(f'X̄ Control Chart\n{product_name} / {machine_name}', fontsize=14)
        ax_xbar.set_xlabel('Time', fontsize=12)
        ax_xbar.set_ylabel('X̄ Thickness', fontsize=12)
        ax_xbar.legend(fontsize=10)
        ax_xbar.tick_params(axis='x', rotation=45)
        ax_xbar.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout() # Adjust layout
        st.pyplot(fig_xbar)

    with chart_col2:
        st.markdown("#### Distribution of X̄ (Histogram)")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 5)) # Consistent naming

        # Plot Histogram
        ax_dist.hist(result_df['X̄'], bins=10, color='skyblue', edgecolor='black', alpha=0.7, label='X̄ Distribution (Frequency)')
        
        # Plot USL, LSL, Target CL
        ax_dist.axvline(usl, color='red', linestyle=':', linewidth=1.5, label=f'USL ({usl:.3f})')
        ax_dist.axvline(lsl, color='red', linestyle=':', linewidth=1.5, label=f'LSL ({lsl:.3f})')
        ax_dist.axvline(cl_target, color='green', linestyle='--', linewidth=1.5, label=f'Target CL ({cl_target:.3f})')
        
        # Calculate mean and std dev of X-bar for normal curve
        mean_xbar = result_df['X̄'].mean()
        std_dev_xbar = result_df['X̄'].std()

        if pd.notna(std_dev_xbar) and std_dev_xbar > 0 and len(result_df['X̄']) > 1:
            # Generate points for the normal distribution curve
            x_vals = np.linspace(min(result_df['X̄'].min(), lsl) - std_dev_xbar, 
                                 max(result_df['X̄'].max(), usl) + std_dev_xbar, 
                                 200)
            pdf_vals = (1 / (std_dev_xbar * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_vals - mean_xbar) / std_dev_xbar)**2)
            
            # Scale PDF to match histogram frequency
            # Get bin counts and edges to calculate bin_width accurately
            counts, bin_edges = np.histogram(result_df['X̄'], bins=10)
            if len(bin_edges) > 1:
                bin_width = bin_edges[1] - bin_edges[0]
            else: # Fallback if only one unique X_bar value (though std_dev_xbar > 0 should prevent this)
                bin_width = (result_df['X̄'].max() - result_df['X̄'].min()) / 10 if result_df['X̄'].max() != result_df['X̄'].min() else 1

            N = len(result_df['X̄'])
            scale_factor = N * bin_width
            
            ax_dist.plot(x_vals, pdf_vals * scale_factor, color='mediumblue', linestyle='--', linewidth=1.5, label='Fitted Normal Curve')
            
            # Plot Mean of X-bar
            ax_dist.axvline(mean_xbar, color='orange', linestyle='-', linewidth=1.5, label=f'Mean X̄ ({mean_xbar:.3f})')
            
            # Plot ±1, 2, 3 Sigma lines from Mean X-bar
            sigma_colors = ['#FFD700', '#FFB000', '#FF8C00'] # Gold, LightOrange, DarkOrange
            for s in range(1, 4):
                label_sigma = f'Mean X̄ ±{s}σ' if s == 1 else None # Label only first set for cleaner legend
                ax_dist.axvline(mean_xbar + s * std_dev_xbar, color=sigma_colors[s-1], linestyle=':', linewidth=1, label=label_sigma if s==1 else f"Mean X̄ ±{s}σ")
                ax_dist.axvline(mean_xbar - s * std_dev_xbar, color=sigma_colors[s-1], linestyle=':', linewidth=1)
        elif pd.notna(mean_xbar): # If std_dev is 0 or NaN, just plot the mean line
             ax_dist.axvline(mean_xbar, color='orange', linestyle='-', linewidth=1.5, label=f'Mean X̄ ({mean_xbar:.3f})')
             st.markdown("ℹ️ _Normal distribution curve and sigma lines cannot be plotted for the Distribution Chart (e.g., standard deviation is zero or insufficient unique data points)._")


        ax_dist.set_title(f'Distribution of X̄ (Average Thickness)\n{product_name} / {machine_name}', fontsize=14)
        ax_dist.set_xlabel('X̄ Thickness', fontsize=12)
        ax_dist.set_ylabel('Frequency', fontsize=12)
        ax_dist.legend(fontsize=10)
        ax_dist.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout() # Adjust layout
        st.pyplot(fig_dist) # THIS WAS THE CRITICAL MISSING LINE FOR THE DISTRIBUTION CHART

    st.markdown("---")
    # Display the processed data table
    st.subheader("Processed Data Table")
    st.dataframe(result_df.style.apply(
        lambda row: ['background-color: #FFCDD2' if row['Out of Spec'] else '' for _ in row], axis=1 # Highlight OOS rows
    ).apply(
        lambda row: ['color: red; font-weight: bold' if row['Rule Violations'] else '' for _ in row], axis=1 # Highlight rule violation text
    ))


    # Function to convert DataFrame to Excel
    @st.cache_data # Cache the conversion
    def convert_df_to_excel(df_to_convert):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_to_convert.to_excel(writer, index=False, sheet_name='SPC_Analysis')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_df_to_excel(result_df)

    st.download_button(
        label="⬇️ Download Results as Excel",
        data=excel_data,
        file_name=f'SPC_Analysis_{product_name.replace(" ", "_")}_{machine_name.replace(" ", "_")}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

else:
    st.info("ℹ️ Enter data and click 'Generate Control Chart & Analysis' to see the results.")
