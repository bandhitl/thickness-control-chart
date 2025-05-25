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
    return data

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

product_name = st.text_input("Product Name:")
machine_name = st.text_input("Machine Name:")

num_rows = st.number_input("Number of time points:", min_value=1, max_value=50, value=10, step=1)

usl = st.number_input("Upper Specification Limit (USL):", value=2.60)
lsl = st.number_input("Lower Specification Limit (LSL):", value=2.40)
cl = (usl + lsl) / 2

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
    st.markdown(f"**Product:** {product_name}  |  **Machine:** {machine_name}")
    st.dataframe(df)

    result = calculate_spc(df.copy())
    cp, cpk = calculate_cp_cpk(result, usl, lsl)

    result['Out of Spec'] = (result['X̄'] > usl) | (result['X̄'] < lsl)

    st.subheader("Process Capability")
    st.write(f"Cp = {cp:.3f}")
    st.write(f"Cpk = {cpk:.3f}")

    if cp < 1.00 or cpk < 1.00:
        st.error("🚨 Cp or Cpk is below 1.00. The process is not capable.")
    elif cp < 1.33 or cpk < 1.33:
        st.warning("⚠️ Cp or Cpk is below 1.33. The process may not be reliable.")
    else:
        st.success("✅ Cp and Cpk are acceptable.")

    if result['Out of Spec'].any():
        st.error("🔴 Some data points are outside the specification limits (USL/LSL). Please review.")

    st.subheader("X̄ Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result['Time'], result['X̄'], marker='o', label='X̄')
    ax.axhline(cl, color='green', linestyle='-', label='CL (Target)')
    ax.axhline(usl, color='purple', linestyle=':', label='USL')
    ax.axhline(lsl, color='purple', linestyle=':', label='LSL')

    for i, row in result.iterrows():
        if row['Out of Spec']:
            ax.plot(row['Time'], row['X̄'], 'ro')

    ax.set_title(f'X̄ Control Chart - {product_name} / {machine_name}')
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
