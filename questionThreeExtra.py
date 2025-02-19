import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_encoded = pd.get_dummies(df, columns=["Class"], drop_first = True)
df_encoded.iloc[:,:8] = df_encoded.iloc[:,:8].astype(int)
df_encoded = df_encoded.to_numpy()


mm = multi_mean(df_encoded)
mm

cov = covMatrix(df_encoded)
print(cov)


# range normalization
def min_max_normalize(df):
    df = pd.DataFrame(df)  # Ensure input is a DataFrame
    return pd.DataFrame((df - df.min()) / (df.max() - df.min()), columns=df.columns)

# Z-score normalization
def z_score_normalize(df):
    df = pd.DataFrame(df)  # Ensure input is a DataFrame
    return pd.DataFrame((df - df.mean()) / df.std(), columns=df.columns)

# Normalize
df_range_normalized = min_max_normalize(df_encoded)
df_zscore_normalized = z_score_normalize(df_encoded)

# Compute covariance and correlation matrices 
cov_matrix = pd.DataFrame(covMatrix(df_range_normalized), index=df_range_normalized.columns, columns=df_range_normalized.columns)
corr_matrix = df_zscore_normalized.corr(method=correlationCoEf)

largest_cov = cov_matrix.unstack().drop_duplicates().sort_values(ascending=False)
largest_cov_pair = largest_cov.iloc[1]  # Skipping first (self-covariance)
largest_cov_features = largest_cov.index[1]

print("Largest estimated covariance: {largest_cov_pair} between {largest_cov_features}")

plt.scatter(df_range_normalized[largest_cov_features[0]], df_range_normalized[largest_cov_features[1]], alpha=0.5)
plt.xlabel(largest_cov_features[0])
plt.ylabel(largest_cov_features[1])
plt.title("Scatter Plot of Range-Normalized Attributes with Largest Covariance")
plt.show()

highest_corr = corr_matrix.unstack().drop_duplicates().sort_values(ascending=False)
highest_corr_pair = highest_corr.iloc[1] 
highest_corr_features = highest_corr.index[1]

print("Largest correlation: {highest_corr_pair} between {highest_corr_features}")

# Scatter plot z-score normalized high
plt.scatter(df_zscore_normalized[highest_corr_features[0]], df_zscore_normalized[highest_corr_features[1]], alpha=0.5)
plt.xlabel(highest_corr_features[0])
plt.ylabel(highest_corr_features[1])
plt.title("Scatter Plot of Z-Score Normalized Attributes with Largest Correlation")
plt.show()

lowest_corr = corr_matrix.unstack().drop_duplicates().sort_values()
lowest_corr_pair = lowest_corr.iloc[0] 
lowest_corr_features = lowest_corr.index[0]

print("Smallest correlation: {lowest_corr_pair} between {lowest_corr_features}")

# Scatter plot for z-score normalized low
plt.scatter(df_zscore_normalized[lowest_corr_features[0]], df_zscore_normalized[lowest_corr_features[1]], alpha=0.5)
plt.xlabel(lowest_corr_features[0])
plt.ylabel(lowest_corr_features[1])
plt.title("Scatter Plot of Z-Score Normalized Attributes with Smallest Correlation")
plt.show()

high_corr_pairs = (corr_matrix.abs() >= 0.5).sum().sum() / 2  # Dividing by 2 to avoid double counting
print("Number of feature pairs with correlation ≥ 0.5: {int(high_corr_pairs)}")

negative_cov_pairs = (cov_matrix < 0).sum().sum() / 2  # Dividing by 2 to avoid double counting
print("Number of feature pairs with negative covariance: {int(negative_cov_pairs)}")
