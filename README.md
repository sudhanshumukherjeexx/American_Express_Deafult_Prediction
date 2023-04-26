# American_Express-Deafult_Prediction
This repository based on American Express Default Prediction which is part of my CIS 530 Advanced Data Mining.

# To Run this code:

### Clone this repository.
### visit inside this folder.
### run this command: `streamlit run app.py` 


### The Descritpion of our Code and Data is as Follows:

* Dataset Details: Drive Link
    1. train.parquet - This file contains all the features with CustomerID.
    2. train_labels.csv - This file contains "Target Labels" for the features present in the 'train.parquet' along with CustomerID. 

* Code Description: GitHub Link

- app.py : This file contains the code which intializes the Dataset, Handles Missing Values, extarct 100,000 rows and Creates a data state and make it accessible for a user for further analysis.

- Pages - This folder contains all code for differet pages available in our python GUI.


                01_🤖 Data Stats.py : You can visit this page to check all the Data Statistics related to our Dataset.
                02_👻 Exploratory Data Analysis.py : To view visualized details of our model, please visit this page.
                03_🥷 Distribution S (Spend) Variable.py : You can view all the details related to Spend variable on this page.
                04_🧙🏻‍♂️ Distribution P (Payment) Variable.py : You can view all the details related to Payment variable on this page.
                05_👨‍🚀 Distribution B (Balance) Variable.py : You can view all the details related to Balance variable on this page.
                06_🕴 Distribution R (Risk) Variable.py : You can view all the details related to Risk variable on this page.
                07_🏂🏼 Distribution D (Deliquency) Variable.py : You can view all the details related to Deliquency variable on this page.
                08_📊 Distribution of Categorical Variable.py : You can view all the details related to Categorical variable on this page.
                09_👨🏻‍💻 CatBoost Model.py - You can view all the details of CatBoost Classifier Model on this page including Model Performance and Stats. When you visit                 this page model is live fetched in background.
                11_👨🏻‍💻 XgBoost Model.py - You can view all the details of XGBoost Classifier Model on this page including Model Performance and Stats. When you visit                   this page model is live fetched in background.
                12_👨🏻‍💻Logistic Regression.py - You can view all the details of Logistic Regression Model on this page including Model Performance and Stats. When you                   visit this page model is live fetched in background.
                13_🕵🏻‍♂️Test our Model.py - You can test our models on this page. This page has slider to select features, intial value is set to the minimum value                       present in the feature.
                
- CatBoost.json : CatBoost Classifier Model is saved in the .JSON format in this file. 
- XGBoost.json : XGBoost Classifier Model is saved in the .JSON format in this file.  
