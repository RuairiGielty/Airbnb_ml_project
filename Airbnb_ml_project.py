import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn import linear_model
from sklearn.linear_model import Ridge
import random
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor
from functools import partial
from sklearn.dummy import DummyRegressor
import re


def normalise(list):
    mean = np.mean(list)
    std = np.std(list)
    for i in range(len(list)):
        list[i] = (list[i] - mean)/std

def setClassValues(price, ratings):
    y = np.array([1] * len(ratings))
    for i in range(len(ratings)):
        if ratings[i] <= 90 and price[i] <= 150:
            y[i] = -1
    return y

def getPoly(degree, X, targetVal):
    poly = PolynomialFeatures(degree)
    fit = poly.fit_transform(X)
    return fit

def set_review_scores_NaN(ratings):
    booleanRatings = np.isnan(ratings)
    for i in range(len(booleanRatings)):
        if booleanRatings[i] == True:
            ratings[i] = 0
    average = ratings.mean()
    for j in range(len(ratings)):
        if ratings[j] == 0:
            ratings[j] = average
    return ratings

def set_NaN_to_zero(vector):
    mask = np.isnan(vector)
    for i in range(len(mask)):
        if mask[i] == True:
            vector[i] = 0
    return vector

def featureCrossVal(features, target):
    model = LinearRegression().fit(features, target)
    scores = cross_val_score(model, features, target, cv = 10, scoring ="neg_mean_squared_error")
    return np.negative(scores.mean()),np.negative(scores.std())

def ridge_crossval_C(feature_matrix, target_vector, c_values):
    mean_arr = []
    std_arr = []
    for c in c_values:
            model = Ridge(alpha = 1/(2*c))
            scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring='neg_mean_squared_error')
            mean_arr.append(np.mean(np.negative(scores)))
            std_arr.append(np.std(np.negative(scores)))
    return mean_arr, std_arr

def ridge_crossval_q(feature_matrix, target_vector, q_values):
        mean_arr = []
        std_arr = []
        for q in q_values:
            poly = PolynomialFeatures(q)
            poly_feature_matrix = poly.fit_transform(feature_matrix)
            model = Ridge()
            scores = cross_val_score(model, poly_feature_matrix, target_vector, cv=5, scoring='neg_mean_squared_error')
            mean_arr.append(np.mean(np.negative(scores)))
            std_arr.append(np.std(np.negative(scores)))
        return mean_arr, std_arr

def lasso_crossval_C(feature_matrix, target_vector, c_values):
    mean_arr = []
    std_arr = []
    for c in c_values:
            model = Lasso(alpha = 1/(2*c))
            scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring='neg_mean_squared_error')
            mean_arr.append(np.mean(np.negative(scores)))
            std_arr.append(np.std(np.negative(scores)))
    return mean_arr, std_arr

def knn_crossval_n(feature_matrix, target_vector, num_neighbours):
    mean_arr = []
    std_arr = []
    for n in num_neighbours:
            model = KNeighborsRegressor(n_neighbors=n)
            scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring='neg_mean_squared_error')
            mean_arr.append(np.mean(np.negative(scores)))
            std_arr.append(np.std(np.negative(scores)))
    return mean_arr, std_arr

def LinReg(features, target, review_score, super_host, facilities):
    model = LinearRegression().fit(features, target)

    #lineIntercept = -((model.intercept_ *1) + (model.coef_[0][0] * review_score) + (model.coef_[0][2] * facilities))/model.coef_[0][1]
    plt.scatter(features[:,0], target, s = 2**2)
    plt.plot(features[:,0], model.predict(features), color = 'red', linewidth = 0.5)
    plt.show()

def removeOutliers(prices, review_scores, is_superhost, amenities):
    new_scores = []
    new_host = []
    new_amenities = []
    new_prices = []
    for i in range(len(prices)):
        if prices[i] < 1999 or review_scores[i] == 74.37914512671794 :
            new_prices.append(prices[i])
            new_scores.append(review_scores[i])
            new_host.append(is_superhost[i])
            new_amenities.append(amenities[i])
    return new_prices, new_scores, new_host, new_amenities

def plot_error(cVals, mean, std, colors, lineColor, title, xLabel, yLabel):
    plt.errorbar(cVals, mean, yerr=std, ecolor=colors, color = lineColor)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    leg_point0 = plt.Rectangle((0,0), 1, 1, fc = colors)
    leg_point1 = plt.Rectangle((0,0), 1,1 , fc = lineColor)
    plt.legend([leg_point0,leg_point1], ["Standard Deviation", "MSE mean"])   
    plt.show()

def classify_feature_matrix(model,feature_matrix):
    return np.sign(model.intercept_ + (model.coef_[0,0]*feature_matrix[:,0]) + (model.coef_[0,1]*feature_matrix[:,1]) 
                   + (model.coef_[0,2]*feature_matrix[:,2]) + (model.coef_[0,3]*feature_matrix[:,3]) + (model.coef_[0,4]*feature_matrix[:,4])
                   + (model.coef_[0,5]*feature_matrix[:,5]) + (model.coef_[0,6]*feature_matrix[:,6]))

def main():
    q = np.array([1,2,3,4])
    
    #dublin_listings = np.genfromtxt("C:/Users/ruair/Documents/4thYear/ml/assignments/group_assignment/datasets/airbnb/dublin_listings.csv",delimiter=',')
    dublin_listings = pd.read_csv("dublin_listings.csv",delimiter=',')


    #EXTRACT FEATURES FROM CSV

    #prices column to numpy array
    prices_pd = pd.to_numeric(dublin_listings['price'], errors='coerce')
    prices = prices_pd.to_numpy()
    np.reshape(prices,(-1,1))
    prices = prices[:,np.newaxis]
    prices = prices.astype(np.float32)
    normalise(prices)

    #host_is_superhost string column to numpy array of bools
    is_superhost = dublin_listings['host_is_superhost'].to_numpy()
    is_superhost[is_superhost == 't'] = 1
    is_superhost[is_superhost == 'f'] = 0
    np.reshape(is_superhost,(-1,1))
    is_superhost = is_superhost[:,np.newaxis]
    is_superhost = is_superhost.astype(np.float32)
    normalise(is_superhost)

    #review_scores_rating to numpy array
    review_scores = dublin_listings['review_scores_rating'].to_numpy()
    np.reshape(review_scores,(-1,1))
    review_scores = review_scores[:,np.newaxis]
    review_scores = set_review_scores_NaN(review_scores)
    review_scores = review_scores.astype(np.float32)
    normalise(review_scores)

    #amenities to numpy array storing number of amenities for each listing
    amenities_pd = dublin_listings['amenities']
    #arrays are stored as strings; need to convert to np array of counts
    amenities = np.zeros(len(dublin_listings))
    for x in range(len(amenities_pd)):
        amenities[x] = len(np.array((amenities_pd[x].replace('[','').replace(']','')).split(',')))
    amenities.astype(int)
    #reshape to be compatible with sklearn functions
    np.reshape(amenities,(-1,1))
    amenities = amenities[:,np.newaxis]
    amenities = amenities.astype(np.float32)
    normalise(amenities)

    #bed count to numpy array
    beds = dublin_listings['beds'].to_numpy()
    np.reshape(beds,(-1,1))
    beds = beds[:,np.newaxis]
    beds = set_NaN_to_zero(beds)
    beds = beds.astype(np.float32)
    normalise(beds)

    #bedroom count to numpy array
    bedrooms = dublin_listings['bedrooms'].to_numpy()
    np.reshape(bedrooms,(-1,1))
    bedrooms = bedrooms[:,np.newaxis]
    bedrooms = set_NaN_to_zero(bedrooms)
    bedrooms = bedrooms.astype(np.float32)
    normalise(bedrooms)

    #accomodates count to numpy array
    accommodates = dublin_listings['accommodates'].to_numpy()
    np.reshape(accommodates,(-1,1))
    accommodates = accommodates[:,np.newaxis]
    accommodates = set_NaN_to_zero(accommodates)
    accommodates = accommodates.astype(np.float32)
    normalise(accommodates)

    #bathroom count to numpy array
    bathrooms_pd = dublin_listings['bathrooms_text']
    bathrooms = np.array(bathrooms_pd)
    bathroom = np.array([0] * len(bathrooms_pd), dtype=float)
    for x in range(len(bathrooms_pd)):
       bathrooms[x] = re.findall(r'[\d.\d]+', str(bathrooms[x]))
       if not bathrooms[x]:
           bathrooms[x] = '1'
       [bath] = bathrooms[x]
       bathroom[x] = bath
    normalise(bathroom)

    print("price: " +str(prices[2]) +"\nis_superhost: " +str(is_superhost[2]) +"\nreview_scores: " +str(review_scores[2])
            +"\namenities: " +str(amenities[2]) +"\nbeds: " +str(beds[2]) +"\nbedrooms: " +str(bedrooms[2])
                    +"\naccomodates: " +str(accommodates[2]) +"\nbathrooms: " +str(bathroom[2]) + "\n\n")



    #CREATE DIFFERENT COMBINATIONS OF FEATURES   
    review_scores + is_superhost
    features_2 = np.column_stack((review_scores, is_superhost))
    #review_scores + is_superhost + amenities
    features_3 = np.column_stack((review_scores, is_superhost, amenities))
    #review_scores + is_superhost + amenities + beds
    features_4 = np.column_stack((review_scores, is_superhost, amenities, beds))
    #review_scores + is_superhost + amenities + beds + bedrooms
    features_5 = np.column_stack((review_scores, is_superhost, amenities, beds, bedrooms))
    #review_scores + is_superhost + amenities + beds + bedrooms + accomodates
    features_6 = np.column_stack((review_scores, is_superhost, amenities, beds, bedrooms, accommodates))
    #all features
    X = np.column_stack((review_scores, is_superhost, amenities, beds, bedrooms, accommodates, bathroom))
    



    #LinReg(review_scores, prices, review_scores, is_superhost, amenities, fit)


    #FEATURE SELECTION
    feature_mean_arr = []
    feature_std_arr = []

    mean, std = featureCrossVal(review_scores, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)
    mean, std = featureCrossVal(features_2, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)
    mean, std = featureCrossVal(features_3, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)
    mean, std = featureCrossVal(features_4, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)
    mean, std = featureCrossVal(features_5, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)
    mean, std = featureCrossVal(features_6, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)
    mean, std = featureCrossVal(X, prices)
    feature_mean_arr.append(mean)
    feature_std_arr.append(std)

    #print(feature_mean_arr)
    #print(feature_std_arr)


    

    #MODEL SELECTION

    #create 80-20 train-test split
    xTrain, xTest, yTrain, yTest = train_test_split(X, prices, test_size = 0.2, random_state = 0)

    #Ridge regression varying C
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    ridge_means, ridge_stds = ridge_crossval_C(xTrain,yTrain,c_values)
    plot_error(c_values, ridge_means, ridge_stds, 'grey', 'red', 'MSE of Ridge (varying C)', 'c value', 'mean squared error')

    #Ridge regression varying poly features
    q_values = [1,2,3,4]
    ridge_q_means, ridge_q_stds = ridge_crossval_q(xTrain,yTrain,q_values)
    plot_error(q_values, ridge_q_means, ridge_q_stds, 'grey', 'red', 'MSE of Ridge (varying q)', 'q value', 'mean squared error')

    #lasso regression varying C
    lasso_means, lasso_stds = lasso_crossval_C(xTrain,yTrain,c_values)
    plot_error(c_values, lasso_means, lasso_stds, 'grey', 'red', 'MSE of Lasso (varying C)', 'c value', 'mean squared error')

    #knn regression varying number of neighbours and gaussian kernel weights
    n_neighbours = [2,5,10,25,50,100,200]
    knn_means, knn_stds = knn_crossval_n(xTrain,yTrain,n_neighbours)
    plot_error(n_neighbours, knn_means, knn_stds, 'grey', 'red', 'MSE of KNN (varying num neighbours)', 'number of neighbours', 'mean squared error')

    #BASELINE - Dummy Regressor
    dummy_model = DummyRegressor(strategy="mean")
    scores = cross_val_score(dummy_model, xTrain, yTrain, cv = 5, scoring = "neg_mean_squared_error")
    


    #print MSE mean and standard deviation of models (varying hyperparameters)
    print("RIDGE MODEL (vary c):")
    print("MSE mean:")
    print("c=0.001: "+ str(ridge_means[0]))
    print("c=0.01: "+ str(ridge_means[1]))
    print("c=0.1: "+ str(ridge_means[2]))
    print("c=1: "+ str(ridge_means[3]))
    print("c=10: "+ str(ridge_means[4]))
    print("c=100: "+ str(ridge_means[5]) + "\n")
    print("MSE standard dev:")
    print("c=0.001: "+ str(ridge_stds[0]))
    print("c=0.01: "+ str(ridge_stds[1]))
    print("c=0.1: "+ str(ridge_stds[2]))
    print("c=1: "+ str(ridge_stds[3]))
    print("c=10: "+ str(ridge_stds[4]))
    print("c=100: "+ str(ridge_stds[5]) + "\n\n")

    print("RIDGE MODEL (vary q):")
    print("MSE mean:")
    print("q=2: " +  str(ridge_q_means[0]))
    print("q=3: " +  str(ridge_q_means[1]))
    print("q=4: " +  str(ridge_q_means[2]) + "\n")
    print("MSE standard dev:")
    print("q=2: " +  str(ridge_q_stds[0]))
    print("q=3: " +  str(ridge_q_stds[1]))
    print("q=4: " +  str(ridge_q_stds[2]) + "\n\n")

    print("LASSO MODEL (vary c):")
    print("MSE mean:")
    print("c=0.001: "+ str(lasso_means[0]))
    print("c=0.01: "+ str(lasso_means[1]))
    print("c=0.1: "+ str(lasso_means[2]))
    print("c=1: "+ str(lasso_means[3]))
    print("c=10: "+ str(lasso_means[4]))
    print("c=100: "+ str(lasso_means[5]) + "\n")
    print("MSE standard dev:")
    print("c=0.001: "+ str(ridge_stds[0]))
    print("c=0.01: "+ str(ridge_stds[1]))
    print("c=0.1: "+ str(ridge_stds[2]))
    print("c=1: "+ str(ridge_stds[3]))
    print("c=10: "+ str(ridge_stds[4]))
    print("c=100: "+ str(ridge_stds[5]) + "\n\n")

    print("KNN MODEL (VARY NUM NEIGHBOURS):")
    print("MSE mean:")
    print("n=2: " + str(knn_means[0]))
    print("n=5: " + str(knn_means[1]))
    print("n=25: " + str(knn_means[2]))
    print("n=50: " + str(knn_means[3]))
    print("n=100: " + str(knn_means[4]))
    print("n=200: " + str(knn_means[5]) + "\n")
    print("MSE standard dev:")
    print("n=2: " + str(knn_stds[0]))
    print("n=5: " + str(knn_stds[1]))
    print("n=25: " + str(knn_stds[2]))
    print("n=50: " + str(knn_stds[3]))
    print("n=100: " + str(knn_stds[4]))
    print("n=200: " + str(knn_stds[5]) + "\n\n")

    print("DUMMY REGRESSOR" +"\n")
    print("MSE mean: " +str(np.negative(scores.mean())))
    print("MSE standard dev: " +str(np.negative(scores.std()))+"\n\n")



    #PREDICTIONS USING TEST DATA
    ridge_model = Ridge(alpha = 1/(2*0.01)).fit(xTrain, yTrain)
    ypred = ridge_model.predict(xTest)
    ridge_mse = mean_squared_error(yTest, ypred)
    print("MSE of ridge model (c=0.01) over test data: " + str(np.mean(ridge_mse)))

    dummy_model = DummyRegressor(strategy="mean").fit(xTrain, yTrain)
    dummy_ypred = dummy_model.predict(xTest)
    dummy_mse = mean_squared_error(yTest, dummy_ypred)
    print("MSE of dummy model over test data: " + str(np.mean(dummy_mse)) + "\n")

    signs = classify_feature_matrix(ridge_model,X)

    # #combining features since they are all normalised to plot on 2-d scatterplot, them multiplication is applying weights to each of them. not sure if it does anything useful
    # combine = (xTrain[:,0] *0.15) + (xTrain[:,1]*0.15) + (xTrain[:,2]*0.15) + (xTrain[:,3]*0.15) + (xTrain[:,4]*0.1)+ (xTrain[:,5] *0.1)+ (xTrain[:,6] *0.1)
    # plt.scatter(combine, yTrain, s = 2**2)
    # plt.show()


if __name__ == "__main__":
    main()
