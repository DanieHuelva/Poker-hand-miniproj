import numpy as np
import pandas as pd



df = pd.read_excel('Raisin_Dataset.xlsx')

# Select only the numeric columns
numeric_data = df.select_dtypes(include=[np.number])

# Convert the numeric DataFrame to a NumPy array
datas = numeric_data.to_numpy()



def find_mean(datas, att1):
    list1 = datas[:, att1]
    sumOfatt = sum(list1)
    mean = sumOfatt/(len(list1))
    return mean


def multi_mean(datas):
    multi_mean = []
    for i in range(datas.shape[1]):  # Iterate through the columns
        column_mean = find_mean(datas, i)
        multi_mean.append([column_mean])  # Append the result as a list
    return np.array(multi_mean)


# print("Multidimensional mean: \n")
# print(multi_mean(datas))
# print()
# print("Numpy multi mean: ", np.mean(datas, axis=0))



def sampleVar(datas, att1):
    list1 = datas[:, att1]            
    mean = find_mean(datas, att1)
    sum2 = 0
    for i in range(len(list1)):
        sum2 += ((list1[i] - mean)**2)       #adding everything for (xi-u)^2
    var = sum2 / (len(list1)-1)          
    return var


# print("sample var: ", sampleVar(datas, 1))      ## getting the sample variance of column 2 which is AxisLength in raisin's data set


def sampleCov(datas, att1, att2):
    list1 = datas[:, att1]
    list2 = datas[:, att2]  
    mean1 = find_mean(datas, att1)
    mean2 = find_mean(datas, att2)
    sum2 = 0
    for i in range(len(list1)):
        sum2 += ((list1[i] - mean1)*(list2[i]-mean2))
    cov = sum2 / (len(list1) -1)
    return cov


# print("sample covariance of Area and Major Axis Length: ", sampleCov(datas, 0, 1))      
#caluclates sample cov between area (attribute 0) and major axis length (attribute 1)


def covMatrix(datas):
    #can use sampleVar and sampleCov
    datas = np.array(datas)
    matrix = []
    for i in range(datas.shape[1]):
        list1 = []
        for j in range(datas.shape[1]):
            if (i == j):                        #for diagonals use samplevar
                cov = sampleVar(datas, i)
                list1.append(cov)
            else:                               #everything else is covariance
                cov = sampleCov(datas, i, j)
                list1.append(cov)
        matrix.append(list1)
    return matrix


# print(covMatrix(datas))



# cov_matrix = covMatrix(datas)

# negative_cov_count = 0

# for i in range(cov_matrix.shape[0]):
#     for j in range(cov_matrix.shape[1]):
#         if i != j and cov_matrix[i, j] < 0:
#             negative_cov_count += 1

# print(f"Number of pairs with negative covariance: {negative_cov_count}")


def correlationCoEf(datas, att1, att2):
    numer = sampleCov(datas, att1, att2)
    o1 = sampleVar(datas, att1) ** 0.5
    o2 = sampleVar(datas, att2) ** 0.5
    return numer / (o1 * o2)


# print(correlationCoEf(datas, 0, 1)) 
#caluclates correlation coeffiecient between area (attribute 0) and major axis length (attribute 1)


# countGreaterFive = 0
# numFeatures = datas.shape[1]

# for i in range(numFeatures):
#     for j in range(i + 1, numFeatures):  
#         correlation = correlationCoEf(datas[:, i], datas[:, j])
        
#         if correlation >= 0.5:
#             countGreaterFive += 1

# print(f"Number of pairs with correlation >= 0.5: {countGreaterFive}")



def standardDev(col, mean):
    variance = sum((x - mean) ** 2 for x in col) / len(col)
    return variance ** 0.5  # Square root of variance


def zNorm(datas):
    columns = list(zip(*datas))

    # Compute mean and standard deviation for each column
    means = [find_mean(datas, i) for i in range(len(datas[0]))]
    stds = [standardDev(col, mu) for col, mu in zip(columns, means)]

    # Normalize data using Z-score formula
    normalized_data = [
        [(x - mu) / sigma if sigma != 0 else 0 for x, mu, sigma in zip(row, means, stds)]
        for row in datas
    ]

    return np.array(normalized_data)



# print("Z score: ", zNorm(datas))


    
    
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
    
    return np.array(normalized_data)

# # Apply Min-Max Normalization
normalized_datas = rangeNorm(datas)
print("our range norm: ", normalized_datas)



def label_encode_2d(inputs):
    label_mappings = {}
    encoded = inputs.copy()
    # Iterate over each column
    for col in range(inputs.shape[1]):
        unique_values, encoded_values = np.unique(inputs[:, col], return_inverse=True)
        encoded[:, col] = encoded_values
        label_mappings[col] = dict(enumerate(unique_values))  # Store mapping

    return encoded.astype(int)


categ = datas[:, -1]   ##the last attribute of our data is categorical
print(label_encode_2d(np.array([categ])))


# # Example usage:
# datas = [
#     ['red', 'small', 'round'],
#     ['blue', 'large', 'square'],
#     ['red', 'large', 'round'],
#     ['green', 'small', 'round'],
#     ['blue', 'small', 'square']
# ]

# encoded_data = label_encode_2d(np.array(datas))

# # Print the encoded data and the label encoders (mappings)
# print("Encoded Data:")
# for row in encoded_data:
#     print(row)

# print("\nLabel Encoders (Mappings):")
# for col_index, encoder in label_encoders.items():
#     print(f"Column {col_index}: {encoder}")



