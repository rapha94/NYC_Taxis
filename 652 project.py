# Data Preprocessing Template

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import scipy.stats as stats

import scipy

import time
from sklearn.decomposition import PCA
from sklearn import decomposition


from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, model_selection

from matplotlib.colors import ListedColormap

import statsmodels.api as sm
from sklearn import linear_model

from matplotlib import style 
style.use('ggplot')

from sklearn.cluster import KMeans


# Importing the dataset

data = pd.read_csv('NYC_taxis.csv')


# Understanding the Data

data.head()
data.tail(10)
sLength = len(data['Pickup Time'])
data['Trip time'] = pd.Series(np.random.randn(sLength), index = data.index)
data['new_col'] = range(0, sLength)
data.info()
data.describe()


#  Dealing with Outliers

def basics_info (initial_numpy_array):
    print("Mean: ", np.mean(initial_numpy_array))
    #median cost of a trip without @ without taxes
    print("Median: ", np.median(initial_numpy_array))
    #The numpy.ptp() function returns the range (maximum-minimum) of values along an axis.
    print("Range: ", np.ptp(initial_numpy_array))
    #standard deviation & variance
    print("SD: ", np.std(initial_numpy_array))
    print("Variance: ", np.var(initial_numpy_array))
    plt.plot(initial_numpy_array)
    plt.show() 

#removing outliers and compute the distribution plot

def removeOutliers(x):
    a = np.array(x)
    outlierConstant = 1
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList


def prob_plot(variable):
    
    mu = np.mean(variable)
    sigma = np.std(variable)

    mu, sigma = mu, sigma
    x = mu + sigma * np.array(variable)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.95)


    plt.xlabel("Prob ")
    plt.ylabel('Probability')
    plt.title('Proba Plot')
   # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([0, 4, 0, 0.05])
    plt.grid(True)
    plt.show()
    
def prob_plot_trip_distance(variable):
    
    mu = np.mean(variable)
    sigma = np.std(variable)

    mu, sigma = mu, sigma
    x = mu + sigma * np.array(variable)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.95)


    plt.xlabel("Prob ")
    plt.ylabel('Probability')
    plt.title('Proba Plot')
   # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([0, 11, 0, 0.6])
    plt.grid(True)
    plt.show()
    
#*******************************************
#STUDY FOR TOTAL AMOUNT 
#*******************************************
total_amount = data.as_matrix(columns = data.columns[19:20])

print("BEFORE")
basics_info(total_amount)
prob_plot(total_amount)

# Remove Outliers
good_total_amount = removeOutliers(total_amount)

print("AFTER")
basics_info(good_total_amount)
prob_plot(good_total_amount)


#*******************************************
#STUDY FOR FARE AMOUNT 
#*******************************************

fare_amount = data.as_matrix(columns=data.columns[17:118])

basics_info(fare_amount)
good_fair_amount = removeOutliers(data[['Fare Amount']])
prob_plot(good_fair_amount)

#*******************************************
#STUDY FOR TIP AMOUNT 
#*******************************************

tip_amount = data.as_matrix(columns=data.columns[14:15])

basics_info(tip_amount)
good_tip_amount = removeOutliers(tip_amount)
basics_info(good_tip_amount)

prob_plot(good_tip_amount)

#*******************************************
#STUDY FOR PASSENGER COUNT
#*******************************************

passenger_count = data.as_matrix(columns=data.columns[5:6])

print(data['Passenger Count'].describe())
basics_info(passenger_count)
good_passenger_count = removeOutliers(passenger_count)


prob_plot(good_passenger_count)
#prob_plot_passenger_count(good_passenger_count)


#*******************************************
#STUDY FOR TRIP DISTANCE
#*******************************************

trip_distance = data.as_matrix(columns=data.columns[6:7])
trip_distance
#print(data["Trip Distance (in miles)"].describe())
print(basics_info(trip_distance))
good_trip_distance = removeOutliers(trip_distance)
#basics_info(good_trip_distance)
prob_plot_trip_distance(good_trip_distance)




#Remove outliers from tip_amount

data['Outlier'] = abs(data['Tip Amount'] - data['Tip Amount'].mean()) > 1.96*data['Tip Amount'].std()
data['Outlier'].value_counts()

a = data[data.Outlier != True]
a

#remove outliers Total Amount
a['Outlier'] = abs(a['Total Amount'] - a['Total Amount'].mean()) > 1.96*a['Total Amount'].std()
a['Outlier'].value_counts()
    
b = a[a.Outlier != True]
b

#remove outliers Total Amount
b['Outlier'] = abs(b['Trip Distance (in miles)'] - b['Trip Distance (in miles)'].mean()) > 1.96*b['Trip Distance (in miles)'].std()
b['Outlier'].value_counts()
    
c = b[b.Outlier != True]
c

#remove outliers Passenger Count
c['Outlier'] = abs(c['Trip Distance (in miles)'] - c['Trip Distance (in miles)'].mean()) > 1.96*c['Trip Distance (in miles)'].std()
c['Outlier'].value_counts()
    
data_without_outliers = c[c.Outlier != True]

data_without_outliers



data[['Total Amount', 'Passenger Count', 'Trip Distance (in miles)', 'Surcharge', 'Tip Amount']].describe()


data_without_outliers[ ['Total Amount', 'Passenger Count', 'Trip Distance (in miles)', 'Surcharge', 'Tip Amount']].describe()
data_without_outliers.describe()

# NEW LENGTH OF DATA

sLength = len(data_without_outliers['Pickup Time'])

# Dummy variables creation
 


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()


#*******************************************
#Dummy variable for Vendor ID
#*******************************************


X = data.iloc[:, 4:5].values

# JUST THE LABELING IS DONE BY LABELENCODER, WE USE ONEHOTENCODER TO DEAL WITH THE ORDER. 

X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


#*******************************************
#Dummy variable for Payment Type
#*******************************************


Y = data.iloc[:, 13:14].values

# JUST THE LABELING IS DONE BY LABELENCODER, WE USE ONEHOTENCODER TO DEAL WITH THE ORDER. 

Y[:,0] = labelencoder_Y.fit_transform(Y[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
Y = onehotencoder.fit_transform(Y).toarray()



#*******************************************
#    REMOVE DUMMY VARIABLE TRAP
#*******************************************
Vendor_ID = X[:,0]
Payment_type = Y[:,0:3]



#*******************************************
#   CONNECTING DUMMY AND ORIGINAL DATA SET
#*******************************************

dt_dummy = pd.DataFrame({'Vendor ID Dummy': Vendor_ID, 'Payment Type Dummy':Payment_type})
                        #'Fare Amount Dummy': c, 'Tip Amount Dummy' : d, 'Total Amount Dummy' : e, 
                        #'Passenger Count Dummy': f, 'Trip Distance Dummy': g, 'Surcharge Dummy': h})
tip_data = data_without_outliers[ ['Total Amount', 'Passenger Count', 'Trip Distance (in miles)', 'Surcharge', 'Tip Amount']]

  len(tip_data) 
len(Vendor_ID)
     
frames = [Vendor_ID,tip_data]

data_dummy = np.concatenate(frames, axis=1)
data_dummy


L = list(data)
A = list(data)
data_frame = pd.DataFrame(tip_data) 
data_frame['Vendor_ID'] = pd.Series(Vendor_ID, index=data_frame.index)



a = np.array(Vendor_ID)

df = pd.DataFrame(Vendor_ID,index= Vendor_ID[:,0])

frames = np.concatenate((Vendor_ID, Payment_type), axis = 1)























































