import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# SPC calculation function
@st.cache_data
def calculate_spc(data):
    data['X̄'] = data[['Thickness1', 'Thickness2', 'Thickness3']].mean(axis=1)
    data['R'] = data[['Thickness1', 'Thickness2', 'Thickness3']].max(axis=1) - data[['Thickness1', 'Thickness2', 'Thickness3']].min(axis=1)

    X_bar_bar = data['X̄'].mean()
    R_bar = data['R'].mean()
    A2 = 1.023  # For sample size = 3

    UCL = X_bar_bar + A2 * R_bar
    LCL = X_bar_bar - A2 * R_bar

    return data, X_bar_bar, R_bar, UCL, LCL

# Cp & Cpk calculation
def calculate_cp_cpk(data, usl, lsl):
    std_dev = data['X̄'].std()
    mean = data['X̄'].mean()
    cp = (usl - lsl) / (6 * std_dev) if std_dev > 0 else np.nan
    cpk = min((usl - mean), (mean - lsl)) / (3 * std_dev) if std_dev > 0 else np.nan
    return cp, cpk

# Streamlit UI
st.title("Thickness Control Chart")
st.markdown("Enter 3 thickness values for each time point and set USL/LSL.")

num_rows = st.number_input("Number of time points:", min_value=1, max_value=50, value=10, step=1)

usl = st.number_input("Upper Specification Limit (USL):", value=2.60)
lsl = st.number_input("Lower Specification Limit (LSL):", value=2.40)

with st.form(key="thickness_form"):
    time_inputs = []
    thickness_inputs = []

    for i in range(num_rows):
        st.markdown(f"### Row {i+1}")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            time_val = st.text_input(f"Time (e.g. 09:00)", key=f"time_{i}", value=f"{9+i:02d}:00")
        with col2:
            t1 = st.number_input("T1", key=f"t1_{i}")
        with col3:
            t2 = st.number_input("T2", key=f"t2_{i}")
        with col4:
            t3 = st.number_input("T3", key=f"t3_{i}")

        time_inputs.append(time_val)
        thickness_inputs.append((t1, t2, t3))

    submit_button = st.form_submit_button(label='🎯 Generate Control Chart')

if submit_button:
    df = pd.DataFrame({
        'Time': time_inputs,
        'Thickness1': [x[0] for x in thickness_inputs],
        'Thickness2': [x[1] for x in thickness_inputs],
        'Thickness3': [x[2] for x in thickness_inputs]
    })
    st.subheader("Input Data")
    st.dataframe(df)

    result, x_bar_bar, r_bar, ucl, lcl = calculate_spc(df.copy())
    cp, cpk = calculate_cp_cpk(result, usl, lsl)

    st.subheader("Process Capability")
    st.write(f"Cp = {cp:.3f}")
    st.write(f"Cpk = {cpk:.3f}")

    if cp < 1.33 or cpk < 1.33:
        st.error("⚠️ Cp or Cpk is below 1.33. The process may not be capable.")
    else:
        st.success("✅ Cp and Cpk are acceptable.")

    st.subheader("X̄ Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result['Time'], result['X̄'], marker='o', label='X̄')
    ax.axhline(x_bar_bar, color='green', linestyle='-', label='CL')
    ax.set_title('X̄ Control Chart')
    ax.set_xlabel('Time')
    ax.set_ylabel('X̄ Thickness')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    def convert_df(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        processed_data = output.getvalue()
        return processed_data

    st.download_button(
        label="⬇️ Download Results",
        data=convert_df(result),
        file_name='SPC_Analysis_Result.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
