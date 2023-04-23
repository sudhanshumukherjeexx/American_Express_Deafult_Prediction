import streamlit as st
import dask.dataframe as dd

#display images and title
st.image("4.png", output_format="auto")
st.title("American Express Fraud Detection")
st.text('This project is based on Kaggle Competition conducted by American Express.\nThe data involved in this competition is around 55 GBs which makes this competition\npart of Big Data. It would be computationally impossible to conduct analysis on \nsuch large data (55 GBs) on a normal laptop even considering having a GPU. As a\nteam we have decided to process the first 100000 rows for our analysis and\nbuilding a model')
st.write('Kaggle Dataset used for this project - [Link](https://www.kaggle.com/competitions/amex-default-prediction/overview)')
 
#Reading Data
new_data = dd.read_parquet('train.parquet')
train_labels = dd.read_csv('train_labels.csv')
data = new_data.merge(train_labels, on='customer_ID')
npart = round(len(data)/100000)
parted_df = data.repartition(npartitions=npart)
data = parted_df.partitions[0]
st.write('🚀 Reading Dataset Completed.')

#Handling Missing Values
missing_values = data.isnull().sum()
missing_count = (missing_values / data.index.size)*100
missing_count = missing_count.compute()

#drop column with missing values greater than 75% missing value count
col_drops = list(missing_count[missing_count >= 75].index)
data = data.drop(col_drops, axis=1)
st.write("🚀 Handling Missing Data")

#Fill missing values
data = data.fillna(data.mean()).compute()
st.write("🚀 Dataset Ready for Further Investigation")

#Defining Categorical Variables
data = data.astype({"B_30": "str", "B_38": "str"})
data = data.astype({'D_114':"str", 'D_116':"str", 'D_117':"str", 'D_120':"str", 'D_126':"str", 'D_63':"str", 'D_64':"str", 'D_66':"str", 'D_68':"str"})

data['S_2'] = dd.to_datetime(data['S_2'])

st.session_state['data'] = data
