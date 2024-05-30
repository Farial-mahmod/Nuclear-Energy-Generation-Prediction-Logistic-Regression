# importing necessary libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# Data source is authentic and taken from https://ourworldindata.org/nuclear-energy . Data may also be put into a .csv file to import and use similary instead of defining like the below approach.

# Initializing numpy arrays to store lightweight data:
years = np.array([ 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
generation = np.array([ 2096, 2112, 2184, 2225, 2322, 2406, 2390, 2431, 2523, 2540, 2613, 2654, 2601, 2719, 2726, 2761, 2703, 2694, 2656, 2725, 2610, 2432, 2448, 2498, 2532, 2571, 2594, 2658, 2754, 2448, 2762, 2639, 2685])

# Defining the threshold value followed by the labels:
threshold = 2532
labels = (generation > threshold).astype(int)

# Splitting the data into train and test:
years_train, years_test, generation_train, generation_test = train_test_split(years, generation, test_size= 0.25)

# Initiating a logistic regression model using sklearn:
model = LogisticRegression()

# Fitting the model:
model.fit(years_train, generation_train)

# LogisticRegression
# LogisticRegression()
# Predicting the generation:
generation_pred = model.predict(years_test)
# Printing the generation prediction:
print(f"Generation Prediction: {generation_pred}")
# output: Generation Prediction: [2685 2685 2685 2685 2685 2685 2685 2685 2685]

# Plotting the visualization using matplotlib:
plt.scatter(years, generation, color='blue')
plt.plot(years, generation, color='red')
plt.xlabel('Year')
plt.ylabel('Nuclear Energy Generation (TWh)')
plt.title('Nuclear Energy Generation Prediction')
plt.show()
