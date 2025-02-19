import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Raisin_Dataset.xlsx')
dfn = df.to_numpy()

numeric_data = df.select_dtypes(include=[np.number])
datas = numeric_data.to_numpy()

df_encoded = pd.get_dummies(df, columns=["Class"], drop_first = True)
df_encoded.iloc[:,:8] = df_encoded.iloc[:,:8].astype(int)
df_encoded = df_encoded.to_numpy()


def find_mean(att1):
    datas = df_encoded
    list1 = datas[:, att1]
    sumOfatt = sum(list1)
    mean = sumOfatt/(len(list1))
    return mean


def multi_mean(datas):
    multi_mean = []
    for i in range(datas.shape[1]):  # Iterate through the columns
        column_mean = find_mean(i)
        multi_mean.append([column_mean])  # Append the result as a list
    return np.array(multi_mean)


def sampleVar(att1):
    datas = df_encoded
    list1 = datas[:, att1]            
    mean = find_mean(att1)
    sum2 = 0
    for i in range(len(list1)):
        sum2 += ((list1[i] - mean)**2)       #adding everything for (xi-u)^2
    var = sum2 / (len(list1)-1)          
    return var



def sampleCov(att1, att2):
    datas = df_encoded
    list1 = datas[:, att1]
    list2 = datas[:, att2]  
    mean1 = find_mean(att1)
    mean2 = find_mean(att2)
    sum2 = 0
    for i in range(len(list1)):
        sum2 += ((list1[i] - mean1)*(list2[i]-mean2))
    cov = sum2 / (len(list1) -1)
    return cov



def covMatrix(datas):
    #can use sampleVar and sampleCov
    datas = np.array(datas)
    matrix = []
    for i in range(datas.shape[1]):
        list1 = []
        for j in range(datas.shape[1]):
            if (i == j):                        #for diagonals use samplevar
                cov = sampleVar(i)
                list1.append(cov)
            else:                               #everything else is covariance
                cov = sampleCov(i, j)
                list1.append(cov)
        matrix.append(list1)
    return matrix


def correlationCoEf(att1, att2):
    numer = sampleCov(att1, att2)
    o1 = sampleVar(att1) ** 0.5
    o2 = sampleVar(att2) ** 0.5
    return numer / (o1 * o2)


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
corr_matrix = df_zscore_normalized.corr(method='pearson')

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
