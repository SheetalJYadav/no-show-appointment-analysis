Medical Appointment No-Show Analysis


This project analyzes over 110,000 medical appointment records  to uncover key factors that influence patient no-shows. It includes data cleaning, exploratory analysis, machine learning, and an interactive dashboard in Power BI.

Tools Used: 1.Python(numpy,panda,sklearn,xgboost,imblearn)
            2.PowerBI
            3.VS Code
            
Machine Learning Model-I have used the XGBClassifier from the XGBoost library to predict patient no-shows. It’s a powerful gradient boosting algorithm known for high performance on structured data.  

Model Performance: 1.Accuracy: ~74%
                   2.F1 Score: Balanced and reliable
                   
Key Performance Indicators (KPIs): 1.Total Appointments: 110,527
                                   2. Show-Up Rate: 79.81%
                                   3. No-Show Rate: 20.19%
                                   4. Average Waiting Days: \~10.17 days
                                   5.Model Accuracy: 74%
                                   
Power BI Dashboard Includes: 1. Pie charts for gender and SMS_received
                             2. Bar and stacked visuals for age, neighborhood
                             3.No-show trend by scheduled day
                             4. KPI cards with summary statistics   
                             
How to Run
1.  Open Code_2.py in Jupyter or VS Code to explore the Python EDA and
 ML code.

2.Launch dashboard/noshow_dash.pbix in Power BI Desktop to view the interactive dashboard.                             
                             
                             
 
