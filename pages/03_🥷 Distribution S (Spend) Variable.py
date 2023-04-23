import streamlit as st
import plotly.express as px

#Dsiplay images and titles
st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

S_columns = [col for col in data.columns if 'S_' in col]
S_columns.append('target')

st.header("Correlation of S (Spend) Variable with the Target Variable:")
corr_s = data[S_columns].corr()
fig = px.imshow(corr_s, text_auto=True, color_continuous_scale='blues')
st.plotly_chart(fig)

st.write("Following S variables has high correlation(>=0.80) between them:\n\nS_3 has correlation of 0.91 with S_7\n\nS_22 has correlation of 0.94 with S_24\n\nFollowing S variables appears to have favorable correlation (>=+-0.25) with target variable:\n\nS_7, S_3")

st.header("Histogram of each S (Spend) Variable:")
for i in S_columns:
    fig = px.histogram(data[i], color=data['target'] ,title=i)
    st.plotly_chart(fig)