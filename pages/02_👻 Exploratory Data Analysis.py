import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import numpy as np

#Display images and title
st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")

#Accessing Session State
data = st.session_state['data']

#Key Metrics Present in the Dataset
mpl.rcParams.update(mpl.rcParamsDefault)
df_types = data.dtypes.value_counts()

fig = plt.figure(figsize=(5,2), facecolor='white')

ax = fig.add_subplot(1,1,1)
font = 'monospace'
ax.text(1, 0.8, "Key Metrics", color='black', fontsize=28, fontweight='bold', fontfamily=font, ha='center')

ax.text(0, 0.4, "{:,d}".format(data.shape[0]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax.text(0, 0.001, "# of rows \nin the dataset", color='dimgrey', fontsize=15, fontweight='light', fontfamily=font, ha='center')

ax.text(0.6, 0.4, "{}".format(data.shape[1]), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax.text(0.6, 0.001, "# of features \nin the dataset", color='dimgrey', fontsize=15, fontweight='light', fontfamily=font, ha='center')

ax.text(1.2, 0.4, "{}".format(len(data.select_dtypes(np.number).columns)), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax.text(1.2, 0.001, "# of numeric columns \nin the dataset", color='dimgrey', fontsize=15, fontweight='light', fontfamily=font, ha='center')

ax.text(1.9, 0.4, "{}".format(len(data.select_dtypes('datetime').columns)), color='#fcba03', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax.text(1.9, 0.001, "# of datecolumns columns \nin the dataset", color='dimgrey', fontsize=15, fontweight='light', fontfamily=font, ha='center')

ax.set_yticklabels('')
ax.tick_params(axis='y',length=0)
ax.set_xticklabels('')
ax.tick_params(axis='x',length=0)

for direction in ['top', 'right', 'left', 'bottom']:
    ax.spines[direction].set_visible(False)

fig.subplots_adjust(top=0.9, bottom=0.2, left=0, hspace=1)

fig.patch.set_linewidth(5)
fig.patch.set_edgecolor('#346eeb')
fig.patch.set_facecolor('#f6f6f6')
ax.set_facecolor('#f6f6f6')
st.header("1. Key Metrics Related to Dataset")
st.pyplot(fig)

#Distribution of Target Variable
tmp_val = data['target'].value_counts()/len(data)*100
tmp = pd.DataFrame()
tmp['target_variables'] = tmp_val.index
tmp['val'] = tmp_val.values
fig = px.pie(tmp, names="target_variables", values="val",color="target_variables",labels = {"target_variables" : "Target Variables","val" : "Percentage [%]"} ,title='Distribution of Target Variable')
fig.update_layout(template="plotly_white")
st.header("2. Distribution of Target Variable")
st.text("Hover on the data to view the exact value")
st.plotly_chart(fig)
st.text("It's clear that our dataset is unbalanced and this is a crucial point to keep in\nmind while modelling. 25% of customers had a default - it will be worth investigating these\ntwo groups separately to find some differences. First let's see how many unique\ncustomers do we have.")

#Unique Customers
st.header("3. Number of Unique Customers")
st.title(f'Number of unique customers: {data["customer_ID"].nunique()}')


#Feature Correlation
st.header("4. Feature Correlation")
correlations = data.corr().abs()
fig = px.imshow(correlations, text_auto=True, color_continuous_scale='Blues', aspect="auto")
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)
st.text("The graph above shows us that most of features are not correlated but there are\nvisible dark 'pixels' meaning we have some highly-correlated ones. Let's breakdown\nthe results, so that we can find the most correlated features.")

#Highly Correlated Variables
unstacked = correlations.unstack()
unstacked = unstacked.sort_values(ascending=False, kind="quicksort").drop_duplicates().head(30)
a = []
for i in range(len(unstacked)): 
    a.append(unstacked.index[i])

b = []
for i in unstacked.values:
    #val=np.log1p(i)
    b.append(i)

c=[]
for i in range(len(unstacked)):
    c.append(i)


df = pd.DataFrame()
df['var'] = a
df['val'] = b
df['rank']= c

st.header("4.1 Top 30 Highly Correlated Features")
fig = px.scatter(df, x='rank', y='val', hover_data=['var','val'], color='var', title="Highly Correlated Variables (rank-wise)",labels={"rank":"Feature Rank in Correlation","val":"Correlation value"})
fig.update_layout(plot_bgcolor='azure')
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)
st.text("The above plot is interactive, hover on the points to view the correlated features and their correlation value.")