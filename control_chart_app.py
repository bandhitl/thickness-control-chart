import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ฟังก์ชันคำนวณ X-bar และ R
@st.cache_data
def calculate_spc(data):
    data['X̄'] = data[['Thickness1', 'Thickness2', 'Thickness3']].mean(axis=1)
    data['R'] = data[['Thickness1', 'Thickness2', 'Thickness3']].max(axis=1) - data[['Thickness1', 'Thickness2', 'Thickness3']].min(axis=1)

    X_bar_bar = data['X̄'].mean()
    R_bar = data['R'].mean()
    A2 = 1.023  # สำหรับ sample size = 3

    UCL = X_bar_bar + A2 * R_bar
    LCL = X_bar_bar - A2 * R_bar

    data['Out of Control'] = (data['X̄'] > UCL) | (data['X̄'] < LCL)

    return data, X_bar_bar, R_bar, UCL, LCL

# UI ด้วย Streamlit
st.title("Thickness Control Chart")
st.markdown("กรอกข้อมูลความหนา (3 ค่า/รอบ) แล้วกดปุ่มเพื่อสร้างกราฟควบคุม")

num_rows = st.number_input("จำนวนรอบการวัด (จำนวนแถว):", min_value=1, max_value=50, value=10, step=1)

with st.form(key="thickness_form"):
    times = [st.text_input(f"เวลา (เช่น 09:00) - แถว {i+1}", value=f"{9+i:02d}:00") for i in range(num_rows)]
    thickness1 = [st.number_input(f"Thickness1 - แถว {i+1}", key=f"t1_{i}") for i in range(num_rows)]
    thickness2 = [st.number_input(f"Thickness2 - แถว {i+1}", key=f"t2_{i}") for i in range(num_rows)]
    thickness3 = [st.number_input(f"Thickness3 - แถว {i+1}", key=f"t3_{i}") for i in range(num_rows)]
    submit_button = st.form_submit_button(label='🎯 สร้าง Control Chart')

if submit_button:
    df = pd.DataFrame({
        'Time': times,
        'Thickness1': thickness1,
        'Thickness2': thickness2,
        'Thickness3': thickness3
    })
    st.subheader("ข้อมูลที่ป้อนเข้ามา")
    st.dataframe(df)

    result, x_bar_bar, r_bar, ucl, lcl = calculate_spc(df.copy())
    st.subheader("ผลการวิเคราะห์ SPC")
    st.dataframe(result)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result['Time'], result['X̄'], marker='o', label='X̄')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(x_bar_bar, color='green', linestyle='-', label='CL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')

    for i, row in result.iterrows():
        if row['Out of Control']:
            ax.plot(row['Time'], row['X̄'], 'ro')

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
        label="⬇️ ดาวน์โหลดผลลัพธ์",
        data=convert_df(result),
        file_name='SPC_Analysis_Result.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
