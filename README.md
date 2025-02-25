AI-Based Prediction of Concrete Strength and Consistency

In this project, multiple AI models are developed to predict the impact of concrete mix proportions on **concrete strength and consistency**. The models used include:  

 **Random Forest Regressor**  

![image](https://github.com/user-attachments/assets/a4ef0b57-0303-46b2-9f9a-bd93ef225271)

   
 **Linear (Lasso) Regressor**  

![image](https://github.com/user-attachments/assets/0b93a407-4398-4eec-aca6-a21fcc3aca4f)

 
 **MLP (Multi-layer Perceptron) Regressor**  

![image](https://github.com/user-attachments/assets/a80e90d7-4624-4758-9d14-5b38ebf2e2ef)

 
 **DNN (Deep Neural Network) and Co-training**  
 
![image](https://github.com/user-attachments/assets/ac54ab32-0e9e-4fdd-816c-de8b3e5c40e3)


### **Data Processing and Analysis**  
To ensure high-quality model inputs, we preprocess the data by **removing outliers using the K-Means method**. We further analyze the relationships between regression variables using **scatter plots and the `sns.pairplot` function**.  
however, in this project, limited info can be get

### **Key Findings**  
Even with advanced models like DNN, **deep preprocessing** plays a crucial role in improving prediction accuracy. We found that introducing **new engineered features**, such as the **water-cement ratio**, as an input variable provides greater predictive improvements than merely fine-tuning hyperparameters.  

This study highlights the importance of **feature engineering** in AI-driven concrete mix design optimization, offering valuable insights for engineers and researchers.
