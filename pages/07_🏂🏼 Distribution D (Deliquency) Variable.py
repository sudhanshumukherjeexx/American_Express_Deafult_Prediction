import streamlit as st
import plotly.express as px

st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

D_columns = [col for col in data.columns if 'D_' in col]
D_columns.append('target')

st.header("Correlation of S (Spend) Variable with the Target Variable:")
corr_d = data[D_columns].corr()
fig = px.imshow(corr_d, text_auto=True, color_continuous_scale='blues')
st.plotly_chart(fig)

st.write("Following D variables has high correlation(>=0.80) between them:\n\nD_42 has correlation of 1.0 with D_110 and D_111\n\nD_58 has correlation of 0.92 with D_74 and 0.93 with D_75\n\nD_62 has correlation of 1.0 with D_77\n\nD_74 has correlation of 0.99 with D_75\n\nD_73 has correlation of -1.0 with D_88\n\nD_139 has correlation of 1.0 with D_143 and D_141\n\nD_141 has correlation of 1.0 with D_143\n\nD_132 has correlation of 0.92 with D_132\n\nD_48 has correlation of 0.84 with D_55 and 0.86 with D_61\n\nD_79 has correlation of 0.89 with D_131\n\nFollowing D variables appears to have favorable correlation (>=+-0.25) with target variable\n\nD_48, D_61, D_44, D_55, D_75, D_58, D_74, D_62, D_77, D_70, D_47, D_42, D_43, D_78, D_41, D_45, D_51")

st.header("Histogram of each D (Delinquency) Variable:")
for i in D_columns:
    fig = px.histogram(data[i], color=data['target'], title=i)
    st.plotly_chart(fig)