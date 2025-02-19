import numpy as np
import pandas as pd



datas = np.array([[0.2, 23, 5.7],
                [0.4, 1, 5.4],
                [1.8, 0.5, 5.2],
                [5.6, 50, 5.1],
                [-0.5, 34, 5.3],
                [0.4, 19, 5.4],
                [1.1, 11, 5.5]])  


##   USE THIS FOR PROB 3
# df = pd.read_excel('Raisin_Dataset.xlsx')

# # Apply one-hot encoding to all categorical columns
# df_encoded = pd.get_dummies(df,columns=["Class"], dtype=int, drop_first=True)  # drop_first=True removes the first dummy column for each categorical column
# df_encoded.to_csv("raisins.csv", index=False)
# datas = df_encoded.to_numpy()


def find_mean(att1):
    sumOfatt = sum(att1)
    mean = sumOfatt/(len(att1))
    return mean


# print("Mean of col 1; ", find_mean(datas[:, 0]))
# print("mean from numpy: ", np.mean(datas[:, 0]))


def multi_mean(datas):
    multi_mean = []
    for i in range(datas.shape[1]):  # Iterate through the columns
        column_sum = np.sum(datas[:, i])  # Sum of each column
        column_mean = column_sum / datas.shape[0]  # Mean of the column
        multi_mean.append([column_mean])  # Append the result as a list
    return multi_mean


# print("Multidimensional mean: \n")
# print(multi_mean(datas))
# print()
# print("Numpy multi mean: ", np.mean(datas, axis=0))


def sampleVar(att1):            
    mean = find_mean(att1)
    sum2 = 0
    for i in range(len(att1)):
        sum2 += ((att1[i] - mean)**2)
    var = sum2 / (len(att1)-1)
    return var



def sampleCov(att1, att2):
    mean1 = find_mean(att1)
    mean2 = find_mean(att2)
    sum2 = 0
    for i in range(len(att1)):
        sum2 += ((att1[i] - mean1)*(att2[i]-mean2))
    cov = sum2 / (len(att1) -1)
    return cov


def covMatrix(datas):
    #can use sampleVar and sampleCov
    datas = np.array(datas)
    matrix = []
    for i in range(datas.shape[1]):
        list1 = []
        for j in range(datas.shape[1]):
            if (i == j):
                cov = sampleVar(datas[:, i])
                list1.append(cov)
            else:
                cov = sampleCov(datas[:,i], datas[:,j])
                list1.append(cov)
        matrix.append(list1)
    return matrix


# print()
# print(covMatrix(datas))

# cov_matrix = np.cov(datas, rowvar=False, ddof=1)
# print("Covariance Matrix:\n", cov_matrix)

cov_matrix = covMatrix(datas)

negative_cov_count = 0

for i in range(cov_matrix.shape[0]):
    for j in range(cov_matrix.shape[1]):
        if i != j and cov_matrix[i, j] < 0:
            negative_cov_count += 1

print(f"Number of pairs with negative covariance: {negative_cov_count}")


def correlationCoEf(att1, att2):
    numer = sampleCov(att1, att2)
    o1 = sampleVar(att1) ** 0.5
    o2 = sampleVar(att2) ** 0.5
    return numer / (o1 * o2)


# print(correlationCoEf(datas[:, 1], datas[:, 2]))
# print()
# corr_matrix = np.corrcoef(datas[:, 1], datas[:, 2])
# print(corr_matrix[0,1])

countGreaterFive = 0
numFeatures = datas.shape[1]

for i in range(numFeatures):
    for j in range(i + 1, numFeatures):  
        correlation = correlationCoEf(datas[:, i], datas[:, j])
        
        if correlation >= 0.5:
            countGreaterFive += 1

print(f"Number of pairs with correlation >= 0.5: {countGreaterFive}")



def standardDev(col, mean):
    variance = sum((x - mean) ** 2 for x in col) / len(col)
    return variance ** 0.5  # Square root of variance


def zNorm(datas):
    columns = list(zip(*datas))
    means = [find_mean(col) for col in columns]
    stds = [standardDev(col, mu) for col, mu in zip(columns, means)]
    normalized_data = [
        [(x - mu) / sigma if sigma != 0 else 0 for x, mu, sigma in zip(row, means, stds)]
        for row in datas
    ]
    
    return normalized_data


# print("Z acore: ", zNorm(datas))
# mean = np.mean(datas, axis=0)
# std_dev = np.std(datas, axis=0, ddof=0)  # Population standard deviation
# # Calculate Z-scores for each column
# z_scores = (datas - mean) / std_dev
# print("numpy's z-score: ", z_scores)
    
    
def rangeNorm(datas):
        # Transpose the data to work column-wise
    columns = list(zip(*datas))

    # Compute min and max for each column
    min_vals = [min(col) for col in columns]
    max_vals = [max(col) for col in columns]

    # Normalize each column
    normalized_data = [
        [(x - min_val) / (max_val - min_val) if max_val != min_val else 0
        for x, min_val, max_val in zip(row, min_vals, max_vals)]
        for row in datas
    ]
    
    return normalized_data


# # Apply Min-Max Normalization
# normalized_datas = rangeNorm(datas)
# print("our range norm: ", normalized_datas)
# min_vals = np.min(datas, axis=0)
# max_vals = np.max(datas, axis=0)

# # Perform range normalization for each column
# normalized_data = (datas - min_vals) / (max_vals - min_vals)

# print("Normalized data (range normalization):")
# print(normalized_data)



def label_encode_2d(datas):
    # Create a dictionary to store label encodings
    label_encoders = {}
    
    # Iterate through each column
    for col_index in range(len(datas[0])):  # For each column
        # Find unique values in the column
        unique_values = set(row[col_index] for row in datas)
        
        # Create a mapping from categorical value to an integer label
        value_to_label = {value: index for index, value in enumerate(unique_values)}
        
        # Store the mapping in the label_encoders dictionary
        label_encoders[col_index] = value_to_label
        
        # Encode the column values
        for row in datas:
            row[col_index] = value_to_label[row[col_index]]
    
    return datas, label_encoders


# # Example usage:
# datas = [
#     ['red', 'small', 'round'],
#     ['blue', 'large', 'square'],
#     ['red', 'large', 'round'],
#     ['green', 'small', 'round'],
#     ['blue', 'small', 'square']
# ]

# encoded_data, label_encoders = label_encode_2d(datas)

# # Print the encoded data and the label encoders (mappings)
# print("Encoded Data:")
# for row in encoded_data:
#     print(row)

# print("\nLabel Encoders (Mappings):")
# for col_index, encoder in label_encoders.items():
#     print(f"Column {col_index}: {encoder}")



