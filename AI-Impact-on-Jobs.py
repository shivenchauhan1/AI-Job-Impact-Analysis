#                       Course: INT375

#                    AI JOB IMPACT DATASET


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# STEP 1: LOAD THE DATASET


df = pd.read_csv("ai_job_impact.csv")
print("Data Loaded Successfully\n")
print("First 10 rows:\n", df.head(10), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
print("DataFrame Info:\n")
df.info()
print("\nDescriptive Statistics:\n", df.describe(), "\n")
print("Shape of DataFrame:\n", df.shape, "\n")
print("Column Names:\n", df.columns, "\n")


# STEP 2: CHECKING MISSING VALUES

print("Missing values in each column:\n", df.isnull().sum(), "\n")

# Fill missing values
df_filled = df.fillna({
    'Age': df['Age'].mean(),
    'Years_Experience': df['Years_Experience'].median(),
    'Salary_Before_AI': df['Salary_Before_AI'].mean(),
    'Salary_After_AI': df['Salary_After_AI'].mean(),
    'Work_Hours_Per_Week': df['Work_Hours_Per_Week'].median(),
    'Job_Satisfaction': df['Job_Satisfaction'].median(),
    'Industry': df['Industry'].mode()[0],
    'Job_Status': df['Job_Status'].mode()[0],
    'Remote_Work': df['Remote_Work'].mode()[0]
})
print("After filling missing values:\n", df_filled.head(), "\n")


# STEP 3: REMOVE DUPLICATES


print("Duplicate rows before removal:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after removal:", df.duplicated().sum(), "\n")


# STEP 4: OUTLIER DETECTION AND HANDLING


numeric_cols = ['Age', 'Years_Experience', 'Salary_Before_AI',
                'Salary_After_AI', 'Work_Hours_Per_Week',
                'Job_Satisfaction', 'Productivity_Change_%']

# IQR Method
print("IQR Method")
df_clean = df.copy()

for col in numeric_cols:
    Q1 = np.percentile(df_clean[col], 25)
    Q3 = np.percentile(df_clean[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    print(f"{col}: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, Outliers={len(outliers)}")
    df_clean[col] = df_clean[col].apply(
        lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x)
    )



# STEP 5: GROUP BY AND PIVOT TABLE


grouped_industry = df_clean.groupby('Industry')['Salary_After_AI'].mean()
print("Mean Salary After AI by Industry:\n", grouped_industry, "\n")

grouped_risk = df_clean.groupby('Automation_Risk')['Job_Satisfaction'].mean()
print("Mean Job Satisfaction by Automation Risk:\n", grouped_risk, "\n")

pivot_table = df_clean.pivot_table(
    values='Salary_After_AI',
    index='AI_Adoption_Level',
    columns='Gender',
    aggfunc='mean'
)
print("Pivot Table - Mean Salary After AI by AI Adoption & Gender:\n", pivot_table, "\n")



# STEP 6: SAVE CLEANED DATASET


df_clean.to_csv("AI_Job_Impact_Cleaned.csv", index=False)
print("Cleaned data saved to: AI_Job_Impact_Cleaned.csv\n")



# OBJECTIVES


# OBJECTIVE 1 - Visualization (Pie Chart)
# Question: What is the proportion of job status among employees after AI adoption?
# Attribute: Job_Status


print("OBJECTIVE 1: Job Status Distribution")


job_status_count = df_clean['Job_Status'].value_counts()
print("Job Status Count:\n", job_status_count, "\n")

plt.figure()
plt.pie(job_status_count, labels=job_status_count.index, autopct="%1.1f%%", colors=['steelblue', 'tomato', 'seagreen'])
plt.title("Job Status Distribution After AI Adoption")
plt.show()



# OBJECTIVE 2 - Visualization (Correlation Heatmap)
# Question: Which numeric attributes are most strongly correlated with each other in the dataset?
# Attribute: all numeric columns



print("OBJECTIVE 2: Correlation Heatmap")

numeric_df = df_clean.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print("Correlation Matrix:\n", corr_matrix, "\n")

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()





# OBJECTIVE 3 - Simple Linear Regression
# Question: Does salary before AI predict salary after AI?
# Attributes: Salary_Before_AI (X) -> Salary_After_AI (Y)


print("OBJECTIVE 3: SLR - Salary Before AI vs Salary After AI")


X2 = df_clean[['Salary_Before_AI']]
y2 = df_clean['Salary_After_AI']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mse2 = mean_squared_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print(f"Intercept: {model2.intercept_:.2f}")
print(f"Coefficient: {model2.coef_[0]:.2f}")
print(f"MSE: {mse2:.4f}")
print(f"R2 Score: {r2_2:.4f}\n")

plt.figure()
plt.scatter(X2, y2, color='green', alpha=0.4, label='Actual Data')
plt.plot(X2, model2.predict(X2), color='red', linewidth=2, label='Regression Line')
plt.xlabel("Salary Before AI")
plt.ylabel("Salary After AI")
plt.title("SLR: Salary Before AI vs Salary After AI")
plt.legend()
plt.grid(True)
plt.show()

sample2 = pd.DataFrame([[60000]], columns=['Salary_Before_AI'])
print(f"Predicted Salary After AI for Salary Before AI 60000: {model2.predict(sample2)[0]:.2f}")




# OBJECTIVE 4 - Z-Test
# Question: Is the average productivity change significantly greater than 0 after AI adoption?
# Attribute: Productivity_Change_%
# H0: Mean productivity change = 0
# H1: Mean productivity change > 0 (Right-Tailed)

print("OBJECTIVE 4: Z-Test - Productivity Change vs 0 (Right-Tailed)")

sample_mean_p = df_clean['Productivity_Change_%'].mean()
population_mean_p = 0
population_std_p = df_clean['Productivity_Change_%'].std(ddof=1)
n_p = len(df_clean['Productivity_Change_%'])
alpha = 0.05

standard_error_p = population_std_p / np.sqrt(n_p)
z_score_p = (sample_mean_p - population_mean_p) / standard_error_p
p_value_p = 1 - norm.cdf(z_score_p)

print(f"Sample Mean Productivity Change: {sample_mean_p:.4f}")
print(f"Population Mean (H0): {population_mean_p}")
print(f"Z-Score: {z_score_p:.4f}")
print(f"P-Value: {p_value_p:.4f}")

if p_value_p < alpha:
    print("Conclusion: Reject H0. Productivity has significantly increased after AI adoption.\n")
else:
    print("Conclusion: Fail to reject H0. No significant increase in productivity.\n")


# OBJECTIVE 5 - Visualization (Scatter Plot)
# Question: Is there a relationship between years of experience and salary after AI adoption?
# Attribute: Years_Experience, Salary_After_AI


print("OBJECTIVE 5: Years of Experience vs Salary After AI")
plt.figure()
sns.scatterplot(x='Years_Experience', y='Salary_After_AI', hue='AI_Adoption_Level', data=df_clean)
plt.title("Years of Experience vs Salary After AI")
plt.xlabel("Years of Experience")
plt.ylabel("Salary After AI")
plt.show()
