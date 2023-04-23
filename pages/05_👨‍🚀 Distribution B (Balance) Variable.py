import streamlit as st
import plotly.express as px

st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

data = st.session_state['data']

B_columns = [col for col in data.columns if 'B_' in col]
B_columns.append('target')

st.header("Correlation of B (Balance) Variable with the Target Variable:")
corr_b = data[B_columns].corr()
fig = px.imshow(corr_b, text_auto=True, color_continuous_scale='blues')
st.plotly_chart(fig)

st.write("Following B variables has high correlation(>=0.80) between them:\n\nB_1 has correlation of 0.99 with B_37\n\nB_1 has correlation of 1.0 with B_11\n\nB_2 has correlation of 0.91 with B_33\n\nB_2 has correlation of 0.85 with B_18\n\nB_11 has correlation of 0.99 with B_37\n\nB_12 has correlation of 0.91 with B_13\n\nB_14 has correlation of 0.89 with B_15\n\nB_7 has correlation of 1.0 with B_23\n\nFollowing B variables appears to have favorable correlation (>=+-0.25) with target variable:\n\nB_9, B_18, B_2, B_33, B_7, B_3, B_23, B_4, B_16, B_1, B_37, B_19, B_20, B_22, B_11, B_8, B_17")

st.header("Histogram of each B (Balance) Variable:")
for i in B_columns:
    fig = px.histogram(data[i],color=data['target'] ,title=i)
    st.plotly_chart(fig)