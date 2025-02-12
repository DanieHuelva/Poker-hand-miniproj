import numpy as np


data = np.loadtxt('poker-hand-testing.data', delimiter=',')  # Load data as a list of lists
data_list = data.tolist()  # Convert NumPy array to a Python list

def find_mean(att1):
    sumOfatt = sum(att1)
    mean = sumOfatt/(len(att1))
    return mean


def multi_mean(data_list):
    multi_mean = []
    for i in range(11):
        column_sum = sum(float(row[i]) for row in data_list)  # Convert to list and sum
        list1 = []
        list1.append(column_sum/1000000)
        multi_mean.append(list1)
    return multi_mean


def sampleVar(att1):            
    mean = find_mean(att1)
    sum2 = 0
    for i in range(len(att1)):
        sum2 += ((att1[i] - mean)**2)
    var = sum2 / (len(att1))
    return var


def sampleCov(att1, att2):
    mean1 = find_mean(att1)
    mean2 = find_mean(att2)
    sum2 = 0
    for i in range(len(att1)):
        sum2 += ((att1[i] - mean1)*(att2[i]-mean2))
    cov = sum2 / (len(att1) -1)
    return cov


def covMatrix(data):
    #can use sampleVar and sampleCov
    matrix = []
    for i in range(11):
        list1 = []
        for j in range(11):
            if (i == j):
                cov = sampleVar(data[:, i])
                list1.append(cov)
            else:
                cov = sampleCov(data[:,i], data[:,j])
                list1.append(cov)
        matrix.append(list1)
    return matrix


def correlationCoEf(att1, att2):
    numer = sampleCov(att1, att2)
    o1 = sampleVar(att1) ** 0.5
    o2 = sampleVar(att2) ** 0.5
    return numer / (o1 * o2)


# print("Multidimensional mean: \n")
# print(multi_mean(data_list))
# print()

# print(sampleVar(data[:,2]))
# print()
# column_sample_variance = np.var(data[:, 2], ddof=1)
# print(f"Sample Variance of column 2:", column_sample_variance)
# print()

# print(sampleCov(data[:, 1], data[:, 2]))
# print()
# cov_matrix = np.cov(data[:, 1], data[:, 2], ddof=1)
# sample_covariance = cov_matrix[0, 1]
# print("Sample Covariance between X and Y:", sample_covariance)
    
# print()
# print(covMatrix(data))

# cov_matrix = np.cov(data, rowvar=False, ddof=1)
# print("Covariance Matrix:\n", cov_matrix)

print(correlationCoEf(data[:, 1], data[:, 2]))
print()
corr_matrix = np.corrcoef(data[:, 1], data[:, 2])

# Extract the correlation coefficient between X and Y
correlation_coefficient = corr_matrix[0, 1]
print("Correlation Coefficient between X and Y:", correlation_coefficient)


