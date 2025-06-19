import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os

# Set style for all plots
plt.style.use('default')
colors = ['#2DD4BF', '#14B8A6', '#0D9488', '#0F766E', '#115E59']
sns.set_palette(colors)

# Create static/images directory if it doesn't exist
os.makedirs('static/images', exist_ok=True)

# Read and preprocess the dataset
df = pd.read_csv("loan_approval_dataset.csv")
print("Dataset loaded. Shape:", df.shape)

# Convert numeric columns to float
numeric_cols = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term', 
                ' cibil_score', ' bank_asset_value', ' luxury_assets_value',
                ' residential_assets_value', ' commercial_assets_value']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Data preprocessing
print("Starting data preprocessing...")

# Calculate asset values
df['Movable_assets'] = df[' bank_asset_value'] + df[' luxury_assets_value']
df['Immovable_assets'] = df[' residential_assets_value'] + df[' commercial_assets_value']

# Create display versions first
df['Education_Display'] = df[' education'].copy()
df['Self_Employed_Display'] = df[' self_employed'].copy()
df['Loan_Status_Display'] = df[' loan_status'].copy()

# Then convert categorical variables for modeling
df[' education'] = df[' education'].replace({' Graduate': 1, ' Not Graduate': 0})
df[' self_employed'] = df[' self_employed'].replace({' Yes': 1, ' No': 0})
df[' loan_status'] = df[' loan_status'].replace({' Approved': 1, ' Rejected': 0})

print("Data preprocessing completed.")

# 1. Data Collection Process Flow
plt.figure(figsize=(15, 8))
features = [' no_of_dependents', ' education', ' self_employed', ' income_annum', 
           ' loan_amount', ' loan_term', ' cibil_score', 'Movable_assets', 'Immovable_assets']
plt.bar(range(len(features)), [1]*len(features), color=colors)
plt.xticks(range(len(features)), [f.strip() for f in features], rotation=45, ha='right')
plt.title('Data Features in Loan Prediction Model', fontsize=16, pad=20)
plt.ylabel('Feature Importance')
for i in range(len(features)):
    plt.text(i, 0.5, str(i+1), ha='center', va='center', fontweight='bold', color='white')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/data_collection.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 2. Correlation Matrix with Enhanced Styling
# Select only numeric columns for correlation
numeric_columns = [' no_of_dependents', ' income_annum', ' loan_amount', ' loan_term', 
                  ' cibil_score', 'Movable_assets', 'Immovable_assets']
correlation_matrix = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
           square=True, linewidths=0.5)
plt.title('Feature Correlation Analysis', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('static/images/correlation_matrix.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 3. Model Comparison with Enhanced Styling
models = ['Logistic\nRegression', 'SVM', 'Decision\nTree', 'Random\nForest']
accuracies = [0.82, 0.79, 0.98, 0.96]
plt.figure(figsize=(12, 6))
bars = plt.bar(models, accuracies, color=sns.color_palette("husl", 4))
plt.title('Model Performance Comparison', fontsize=16, pad=20)
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2%}',
             ha='center', va='bottom', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('static/images/model_comparison.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 4. CIBIL Score Analysis
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(data=df, x=' cibil_score', hue='Loan_Status_Display', common_norm=False, 
            fill=True, alpha=0.5)
plt.title('CIBIL Score Distribution by Loan Status', fontsize=12)
plt.xlabel('CIBIL Score')
plt.ylabel('Density')
plt.axvline(x=700, color='red', linestyle='--', alpha=0.5, label='Critical Score')
plt.legend(title='Loan Status')

plt.subplot(1, 2, 2)
sns.boxplot(x='Loan_Status_Display', y=' cibil_score', data=df, palette=colors[:2])
plt.title('CIBIL Score Box Plot by Loan Status', fontsize=12)
plt.xlabel('Loan Status')
plt.ylabel('CIBIL Score')
plt.tight_layout()
plt.savefig('static/images/cibil_analysis.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 5. Income vs Loan Amount Analysis
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
scatter = plt.scatter(df[' income_annum'], df[' loan_amount'], 
                     c=df[' cibil_score'], cmap='viridis', 
                     alpha=0.6, s=100)
plt.colorbar(scatter, label='CIBIL Score')
plt.title('Income vs Loan Amount\n(Color: CIBIL Score)', fontsize=12)
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x=' income_annum', y=' loan_amount', 
                hue='Loan_Status_Display', style='Loan_Status_Display', s=100,
                palette=colors[:2])
plt.title('Income vs Loan Amount\nby Approval Status', fontsize=12)
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/income_loan_analysis.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 6. Asset Distribution Analysis
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Loan_Status_Display', y='Movable_assets', data=df, palette=colors[:2])
plt.title('Movable Assets by Loan Status', fontsize=12)
plt.xticks(rotation=0)
plt.ylabel('Movable Assets Value')

plt.subplot(1, 2, 2)
sns.boxplot(x='Loan_Status_Display', y='Immovable_assets', data=df, palette=colors[:2])
plt.title('Immovable Assets by Loan Status', fontsize=12)
plt.xticks(rotation=0)
plt.ylabel('Immovable Assets Value')
plt.tight_layout()
plt.savefig('static/images/asset_analysis.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 7. Education and Employment Impact
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
education_approval = pd.crosstab(df['Education_Display'], df['Loan_Status_Display'], normalize='index') * 100
education_approval.plot(kind='bar', stacked=True, color=colors[:2])
plt.title('Education Impact on Loan Approval', fontsize=12)
plt.xlabel('Education Level')
plt.ylabel('Percentage')
plt.legend(title='Loan Status')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
employment_approval = pd.crosstab(df['Self_Employed_Display'], df['Loan_Status_Display'], normalize='index') * 100
employment_approval.plot(kind='bar', stacked=True, color=colors[:2])
plt.title('Self-Employment Impact on Loan Approval', fontsize=12)
plt.xlabel('Self Employed')
plt.ylabel('Percentage')
plt.legend(title='Loan Status')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('static/images/education_employment_impact.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()

# 8. Decision Tree Visualization
X = df[[' no_of_dependents', ' education', ' self_employed', ' income_annum', 
        ' loan_amount', ' loan_term', ' cibil_score', 'Movable_assets', 'Immovable_assets']]
y = df[' loan_status']

# Create display names for features
feature_names = ['Dependents', 'Education', 'Self Employed', 'Income', 
                'Loan Amount', 'Loan Term', 'CIBIL Score', 'Movable Assets', 'Immovable Assets']

# Train a small decision tree for visualization
small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
small_tree.fit(X, y)

plt.figure(figsize=(20, 10))
plot_tree(small_tree, feature_names=feature_names,
          class_names=['Rejected', 'Approved'], 
          filled=True, rounded=True, fontsize=10)
plt.title('Simplified Decision Tree Model', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('static/images/decision_tree.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
