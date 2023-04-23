import streamlit as st
import plotly.express as px

st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

R_columns = [col for col in data.columns if 'R_' in col]
R_columns.append('target')

st.header("Correlation of R (Risk) Variable with the Target Variable:")
corr_r = data[R_columns].corr()
fig = px.imshow(corr_r, text_auto=True, color_continuous_scale='blues')
st.plotly_chart(fig)

st.write("Following R variables has high correlation(>=0.80) between them:\n\nR_4 has correlation of 0.79 with R_2\n\nR_5 has correlation of 0.8 with R_8\n\nFollowing R variables appears to have favorable correlation (>=+-0.25) with target variable\n\nR_1, R_3, R_27, R_2")

st.header("Histogram of each R (Risk) Variable:")
for i in R_columns:
    fig = px.histogram(data[i], color=data['target'] ,title=i)
    st.plotly_chart(fig)