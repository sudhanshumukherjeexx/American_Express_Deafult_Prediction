import streamlit as st
import plotly.express as px

st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

P_columns = [col for col in data.columns if 'P_' in col]
P_columns.append('target')

st.header("Correlation of P (Payment) Variable with the Target Variable:")
corr_p = data[P_columns].corr()
fig = px.imshow(corr_p, text_auto=True, color_continuous_scale='blues')
st.plotly_chart(fig)

st.write("Following P variables has high correlation(>=0.80) between them\n\nNone of the P variables appears to be highly correlated.\n\nFollowing P variables appears to have favorable correlation (>=+-0.25) with target variable:\n\nP_2")


st.header("Histogram of each P (Payment) Variable:")
for i in P_columns:
    fig = px.histogram(data[i], color=data['target'], title=i)
    st.plotly_chart(fig)