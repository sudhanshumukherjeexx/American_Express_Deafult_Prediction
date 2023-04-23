import streamlit as st

#display images and titles
st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

#acceessing session state
data = st.session_state['data']

#description
st.header('Data Details')
st.text('The dataset contains aggregated profile features for each customer at each\nstatement date.\nFeatures are anonymized and normalized, and fall into the following\ngeneral categories:')
st.text("D_* = Delinquency Variables")
st.text("S_* = Spend Variables")
st.text("P_* = Payment Variables")
st.text("B_* = Balance Variables")
st.text("R_* = Risk Variables")

#categorical variables
st.text("Following are categorical variables: \n ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']")

#Dataset Details
st.write("Compressed data in Parquet Format for American Express - Default Prediction")
st.write("float64 categories converted to float16")
st.write("int64 categories converted to int8")
st.write("object categories converted to category")
st.write("num cols agg stats -> 'mean', 'std', 'min', 'max', 'last'")
st.write("cat cols agg stats -> 'count', 'last', 'nunique'")

#Data Statistics
st.header('Data Statistics')
st.write(data.describe())

#Data Header
st.header('Data Header')
st.write(data.head())

#Dataset Correlation
st.header('Correlation in Dataset')
st.write(data.corr())