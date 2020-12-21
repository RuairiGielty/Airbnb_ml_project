import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
import random
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
    print(booleanRatings)
    for i in range(len(booleanRatings)):
        if booleanRatings[i] == True:
            ratings[i] = 0
    average = ratings.mean()
    print(average)
    for j in range(len(ratings)):
        if ratings[j] == 0:
            ratings[j] = average
        #print(ratings[j])
    return ratings

def featureCrossVal(features, target):
    model = LinearRegression().fit(features, target)
    scores = cross_val_score(model, features, target, cv = 10, scoring ="neg_mean_squared_error")
    return np.negative(scores.mean()),np.negative(scores.std())

def linearRegCrossVal(features, target, q):
    mean_arr = []
    std_arr = []
    for i in range(len(q)):
        poly = PolynomialFeatures(q[i])
        poly_mat = poly.fit_transform(features)
        model = LinearRegression()
        scores = cross_val_score(model, poly_mat, target, cv = 5, scoring= "neg_mean_squared_error")
        mean_arr.append(np.negative(scores.mean()))
        std_arr.append(np.negative(scores.std()))
    print(mean_arr)
    print(std_arr)
    plt.errorbar(q, mean_arr, yerr = std_arr)
    plt.show()

def ridge_crossval_C(feature_matrix, target_vector, c_values):
    mean_arr = []
    std_arr = []
    for c in c_values:
            model = Ridge(alpha = 1/(2*c))
            scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring='neg_mean_squared_error')
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

def knn_crossval_gamma(feature_matrix, target_vector, num_neighbours, gamma):
    mean_arr = []
    std_arr = []
    for g in gamma:
            model = KNeighborsRegressor(n_neighbors=num_neighbours, weights=partial(gaussian_kernel,g))
            scores = cross_val_score(model, feature_matrix, target_vector, cv=5, scoring='neg_mean_squared_error')
            mean_arr.append(np.mean(np.negative(scores)))
            std_arr.append(np.std(np.negative(scores)))
    return mean_arr, std_arr

def gaussian_kernel(gamma, distances):
    weights = np.exp(-gamma * (distances**2))
    return weights/np.sum(weights)



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

def main():
    q = np.array([1,2,3,4])
    lists = np.array([1, 2, 3, 4])
    feature_mean_arr = np.array([0] * 4)
    feature_std_arr = np.array([0] * 4)
    #dublin_listings = np.genfromtxt("C:/Users/ruair/Documents/4thYear/ml/assignments/group_assignment/datasets/airbnb/dublin_listings.csv",delimiter=',')
    dublin_listings = pd.read_csv("dublin_listings.csv",delimiter=',')

    #prices column to numpy array
    prices_pd = pd.to_numeric(dublin_listings['price'], errors='coerce')
    prices = prices_pd.to_numpy()
    np.reshape(prices,(-1,1))
    prices = prices[:,np.newaxis]

    #host_is_superhost string column to numpy array of bools
    is_superhost = dublin_listings['host_is_superhost'].to_numpy()
    is_superhost[is_superhost == 't'] = 1
    is_superhost[is_superhost == 'f'] = 0
    np.reshape(is_superhost,(-1,1))
    is_superhost = is_superhost[:,np.newaxis]

    #review_scores_rating to numpy array
    review_scores = dublin_listings['review_scores_rating'].to_numpy()
    np.reshape(review_scores,(-1,1))
    review_scores = review_scores[:,np.newaxis]
    #print((review_scores == np.nan))
    #review_scores1 = set_review_scores_NaN(review_scores)
    #ind = np.where(review_scores1 == 74.36976803)
    #print(np.array_equal(review_scores[ind],review_scores1[ind]))

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

    #putting all features in a column stack.
    review_scores = set_review_scores_NaN(review_scores)
    review_superhost_augment = review_scores
    review_superhost_augment[np.logical_not(is_superhost.astype(int))] -= 20

    #prices, review_scores, is_superhost, amenities = removeOutliers(prices, review_scores, is_superhost, amenities)

    #creating different sets of features.
    #review_scores + is_superhost
    rs_and_is = np.column_stack((review_scores, is_superhost))
    #review_scores + amenities
    rs_and_am = np.column_stack((review_scores, amenities))
    #all three
    X = np.column_stack((review_scores, is_superhost, amenities))

    augment = np.column_stack((review_superhost_augment,amenities))
    

    fit = getPoly(1, X, prices)
    xTrain, xTest, yTrain, yTest = train_test_split(X, prices, test_size = 0.2, random_state = 0)
    #linearRegCrossVal(xTrain, yTrain, q)
    #LinReg(review_scores, prices, review_scores, is_superhost, amenities, fit)
    mean, std = featureCrossVal(review_scores, prices)
    feature_mean_arr[0] = mean
    feature_std_arr[0] = std
    mean, std = featureCrossVal(rs_and_is, prices)
    feature_mean_arr[1] = mean
    feature_std_arr[1] = std
    mean, std = featureCrossVal(rs_and_am, prices)
    feature_mean_arr[2] = mean
    feature_std_arr[2] = std
    mean, std = featureCrossVal(review_superhost_augment, prices)
    feature_mean_arr[3] = mean
    feature_std_arr[3] = std

    plot_error(lists, feature_mean_arr, feature_std_arr, 'red', 'grey', 'title', 'xlabel', 'ylabel')
    plt.show()

    linearRegCrossVal(review_superhost_augment, prices, q)

    
    #use review_scores, is_superhost, amenities as features

    #Ridge regression varying C
    c_values = [0.01, 0.1, 1, 10]
    ridge_means, ridge_stds = ridge_crossval_C(X,prices,c_values)

    #lasso regression varying C
    lasso_means, lasso_stds = lasso_crossval_C(X,prices,c_values)

    #knn regression varying number of neighbours and gaussian kernel weights
    n_neighbours = [2,5,10,25,50]
    knn_means, knn_stds = knn_crossval_n(X,prices,n_neighbours)

    gamma = [0,1,5,10,25]
    knn_gamma_means, knn_gamma_stds = knn_crossval_gamma(X,prices,25,gamma)

    #compare MSE mean and standard deviation of models
    print("RIDGE MODEL:")
    print("MSE mean:" + str(ridge_means))
    print("MSE standard dev:" + str(ridge_means) + "\n\n")
    print("LASSO MODEL:")
    print("MSE mean:" + str(lasso_means))
    print("MSE standard dev:" + str(lasso_means) + "\n\n")
    print("KNN MODEL (VARY NUM NEIGHBOURS):")
    print("MSE mean:" + str(knn_means))
    print("MSE standard dev:" + str(knn_means) + "\n\n")
    print("KNN MODEL (VARY GAUSSIAN WEIGHTS):")
    print("MSE mean:" + str(knn_gamma_means))
    print("MSE standard dev:" + str(knn_gamma_means) + "\n\n")
    



    # plt.rc('font', size=14)
    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.scatter(review_scores, prices, 1, color='red')

    # plt.xlabel("price"); plt.ylabel("review score")
    # #plt.legend(["class 1","class -1"], prop={'size' : 10})
    # #plt.title("Dataset " + str(dataset_index) + " plot")
    # plt.show()

if __name__ == "__main__":
    main()


