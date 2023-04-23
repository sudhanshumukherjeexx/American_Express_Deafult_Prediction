import streamlit as st
import plotly.express as px

st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

categorical_columns = ["B_30", "B_38",'D_114','D_116','D_117','D_120','D_126','D_63','D_64','D_66','D_68']
for i in categorical_columns:
    fig = px.pie(names=data[i],values=data[i], title=i)
    st.plotly_chart(fig)