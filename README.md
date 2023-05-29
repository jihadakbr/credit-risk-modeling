
# Credit Risk Modeling - ID/X Partners

As the final task of your internship as a Data Scientist at **ID/X Partners**, this time you will be involved in a project for a lending company. You will collaborate with various other departments in this project to provide technological solutions for the company. You are asked to build a model that can predict credit risk using a dataset provided by the company, which consists of loan data that has been accepted and rejected. Additionally, you also need to prepare visual media to present the solution to the client. Make sure the visual media you create is clear, easy to read, and communicative. You can carry out this end-to-end solution development in your preferred programming language while adhering to the framework/methodology of Data Science.
## Dataset Source

I will use a dataset from [Kaggle](https://drive.google.com/file/d/1wFnz5ozhqX0_FB123bagKWmDRFRF1S_a/view?usp=share_link) that pertains to consumer loans granted from 2007 to 2014 by Lending Club, which is a peer-to-peer lending platform based in the United States.
## Dataset Files

1. loan_data_2007_2014.csv
2. LCDataDictionary.xlsx

Target Variable Description:
* Target variable = 0 → Rejected for a loan → Defaulter
* Target variable = 1 → Accepted for a loan → Non-Defaulter
## Tools

* Programming language: Python.
* Data Tool: Jupyter Notebook.
* Reporting Tool: Microsoft PowerPoint.
## The Project Workflow

#### credit-risk-modeling-using-a-scorecard.ipynb
1. Problem Formulation    
2. Data Collecting
3. Data Understanding
4. Data preprocessing
5. Exploratory Data Analysis (EDA) and Data Visualization
7. Model Selection and Building
8. Scorecard Development

#### credit-risk-modeling-using-various-models.ipynb
1. Problem Formulation    
2. Data Collecting
3. Data Understanding
4. Data preprocessing
5. Exploratory Data Analysis (EDA) and Data Visualization
7. Model Selection and Building

## Results

#### credit-risk-modeling-using-a-scorecard.ipynb
![scorecard development](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiv3_i46-yu90SdIGy4ZbWKaII2YCRpf-saFJrmBqpW62gxKvE5jfpPstivtNEzS8vmTdBGWfnT_naCB90hodlfEmWxUp_3xsxV1mPX-PR0WN2DQJ2NtOlhVFPXIzFyi9XsL5kppd0OD72FKNJWjVl8mfV92VyhB4SQkfVOjUmQyQL2R-4rE4KuT6TO/s1600/scorecard-development.png)

![Money Losses and Saved](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhodbMdbXLHXBJV5JNm9NmH7lVtNmMubtgjaTgqF36Z11fCSIP2928bisimI2DB1DFpVzj6yzOYSu9XjbKN1sWRqHXQY73HANB-Z53DsWXtdGgWSOsJmGFeErc7-GGzhkuSvxj0iBNGu9h9SwolIBzNGSd4rIv3VhXhM0e3DCJvaYEagQJ54pXsDrD0/s1600/money-losses-and-saved.png)

#### credit-risk-modeling-using-various-models.ipynb
![LGBM Classifier](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgLsTw3XDCBeEn2DU95XtgJhG6isgnNZV6DHnw4UTyjPXuAxyxO6XiqOWCJ_1mNSEZFypRcytbdd6PO9eFGLHdgMZ0yQCuWC7rLu5Ic9-0IJu-mX2pdn6Y11_E3R8yAPvBkyb5RlMrkNMTvvNrt9buYexa3Sx5UOjYOPm05PZ5bO8G4hgdqE94IouC0/s1600/lgbm-classifier.png)

![Money Losses and Saved](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiW9tD8wFNjQegb2DmqrHXCbEoKl5uECZW__A2ZI6UnfMPb6yB8janWAuVKNqyvvE2-jRzuxTBM8xE7JBpaDMNT15EraGUgADMqJnKGHmU6lF2p5h-LUv7h-2H_jeKQY9mpR6rTKdGegRoLHpJ9WHmm5nnHfNnGSqhZ_X1yZaf2mO3i9zylN0Mgtrbw/s1600/money-losses-and-saved.png)

## Conclusions

#### credit-risk-modeling-using-a-scorecard.ipynb
* The loan_data_2007_2014.csv file (containing 466,285 rows and 74 columns) contain numerous missing values and outliers, which have been handled using the WOE binning technique.
* No duplicate values are present in the dataset.
* The target variables consist of 89.1% non-defaulters (accepted) and 10.9% defaulters (rejected).
* Feature selection has been performed using Weight of Evidence (WOE) and Information Value (IV).
* Logistic regression was employed in a machine learning model, yielding the following metrics: threshold ≈ 0.22, accuracy ≈ 0.90, precision ≈ 0.93, recall ≈ 0.96, F1 ≈ 0.95, AUROC ≈ 0.84, Gini ≈ 0.67, and AUCPR ≈ 0.97. These metrics are very good for credit risk modeling.
* Consequently, the company is expected to save around 1,000,000,000 USD while incurring a loss of approximately 9,000,000 USD.

#### credit-risk-modeling-using-various-models.ipynb
* The loan_data_2007_2014.csv file (containing 466,285 rows and 74 columns) contains numerous missing values and outliers, which have been handled through data imputation methods, such as using the mean for numerical variables and the mode for categorical variables.
* No duplicate values are present in the dataset.
* The target variables consist of 89.9% non-defaulters (accepted) and 10.1% defaulters (rejected).
* Feature selection has been performed using the Chi-Square Test, ANOVA, and Correlation Matrix.
* Various machine learning models have been implemented on the data, such as logistic regression, ridge classifier, SGD classifier, passive-aggressive classifier, linear discriminant analysis, quadratic discriminant analysis, decision tree, extra tree, ada boost, Gaussian NB, and LGBM classifier.
* The resulting model achieved a higher AUROC score of 0.99 in the LGBM Classifier. This model proceeded further and produced the following metrics: threshold ≈ 0.5, accuracy ≈ 0.95, precision ≈ 0.99, recall ≈ 0.95, F1 ≈ 0.97, AUROC ≈ 0.98, Gini ≈ 0.97, and AUCPR ≈ 0.99. These metrics are very good for credit risk modeling.
* Consequently, the company is expected to save around 1,000,000,000 USD while incurring a loss of approximately 9,000,000 USD.
