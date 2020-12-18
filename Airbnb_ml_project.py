import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dublin_listings = np.genfromtxt("C:/Users/ruair/Documents/4thYear/ml/assignments/group_assignment/datasets/airbnb/dublin_listings.csv",delimiter=',')
dublin_listings = pd.read_csv("dublin_listings.csv",delimiter=',')

#prices column to numpy array
prices_pd = pd.to_numeric(dublin_listings['price'], errors='coerce')
prices = prices_pd.to_numpy()

#host_is_superhost string column to numpy array of bools
is_superhost = dublin_listings['host_is_superhost'].to_numpy()
is_superhost[is_superhost == 't'] = 1
is_superhost[is_superhost == 'f'] = 0

#review_scores_rating to numpy array
review_scores = dublin_listings['review_scores_rating'].to_numpy()

#amenities to numpy array storing number of amenities for each listing
amenities_pd = dublin_listings['amenities']
#arrays are stored as strings; need to convert to np array of counts
amenities = np.zeros(len(dublin_listings))
for x in range(len(amenities_pd)):
    amenities[x] = len(np.array((amenities_pd[x].replace('[','').replace(']','')).split(',')))
amenities.astype(int)


print(prices.shape)
print(amenities.shape)
print(is_superhost.shape)
print(review_scores.shape)

plt.rc('font', size=14)
plt.rcParams['figure.constrained_layout.use'] = True
plt.scatter(prices, review_scores, 1, color='red')

plt.xlabel("price"); plt.ylabel("review score")
#plt.legend(["class 1","class -1"], prop={'size' : 10})
#plt.title("Dataset " + str(dataset_index) + " plot")
plt.show()

