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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor

def setClassValues(price, ratings):
    y = np.array([1] * len(ratings))
    for i in range(len(ratings)):
        if ratings[i] <= 90 and price[i] <= 150:
            y[i] = -1
    return y

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
        print(ratings[j])
    return ratings

def main():
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

    plt.rc('font', size=14)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(prices, review_scores, 1, color='red')

    plt.xlabel("price"); plt.ylabel("review score")
    #plt.legend(["class 1","class -1"], prop={'size' : 10})
    #plt.title("Dataset " + str(dataset_index) + " plot")
    plt.show()

if __name__ == "__main__":
    main()
