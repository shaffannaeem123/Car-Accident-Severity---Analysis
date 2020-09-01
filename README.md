# Car Accident Severity Analysis: Seattle, Washington (IBM Capstone Project)

The Car Accident Severity project aims to understand the effects of various factors on the likelihood and severity of car accidents using a Machine Learning Model.

## Introduction

Road traffic injuries are among the ten leading causes of death worldwide, and they are the leading cause of death among young adults aged 15–29 years. Such accidents also lead to 20–50 million non-fatal injuries, and many people incur a disability as a result of their injury. According to WHO, 1·25 million people worldwide died in road traffic accidents in 2013.  
The world also suffers greatly on an economic front due to road accidents and the costs of these accidents are covered by taxpayer money.  Among all countries, the USA has the largest economic burden of road injuries of $487 billion, followed by China ($364 billion) and India ($101 billion); according to a research journal published by THE LANCET.

## Major Stakeholders

- Travelers
- Insurance Companies
- State Health Department
- Emergency Services
- Infrastructural Development Authorities
- Families of the Travelers
- Taxpayers

## Problem

There is a lack of awareness amongst travelers regarding the risks they might be facing while taking certain routes, crossing certain areas, driving at a specific speed, driving on a specific road, and being inattentive while driving, etc. High-accident-prone areas are seldom inspected with regards to road maintenance, and deployment of additional emergency services personnel, causing additional damage caused by road accidents.

## Goal

This project aims to predict whether an accident that happens under a specific set of circumstances will be an accident limited to *property damage* or if it will include some form of *physical injury* to the driver and/or the passengers.

## Data

The dataset that being used for this project is majorly provided by the government and pertains to the city of Seattle, Washington. It includes observations from 2004 to 2020. The number of observations in the data are enough to formulate a machine learning model. A large majority of the feature-set contains qualitative and categorical data, which is why performing a simple *Multiple Linear Regression* or* Polynomial Regression* is not the good option. The target variable for this model is the *level of severity* of the car accident (property damage only versus physical injury).

This is the dataset in CSV format in case it needs to be viewed: (https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv)

After initial data exploration, we determined the following features to be most relevant when predicting *Accident Severity*.

### Selected Independent/Predictor Variables

![Predictor Variables](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Selected%20Feature%20Variables.PNG)

### Selected Target Variable

- **SEVERITYCODE**: A code that corresponds to the severity of the collision

## Methodology

### Data Collection

The dataset used for this project is a public dataset and illustrates the circumstances in which car accidents take place in Seattle, Washington, from 2004 to 2020.

### Data Cleaning & Transformation

After gaining an understanding of the problem, the data had to be transformed to a form on which a machine learning model could be implemented. The first thing that was done was to check the data types of each variable and then explore how many variables were missing some entries.

The data types in the Seattle dataset mostly comprised *categorical variables* and *objects*; it was concluded that a Simple/Multiple/Polynomial Regression would not work here. The variables in the dataset were listed in plain english and most of them were to be encoded with integers in an *ordinal* manner. 

#### Data-types in the Dataset

![Data Types](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Data%20Types.JPG)

The frequency of datapoints that contained entries that could be readily understood, for example, *Y* for *Yes*, *N* for *No*, or *0* for *False* and *1* for *True*, was higher for some variables than the others. 

#### Frequncies in Dataset Columns before the Data was Transformed

![Data Exploration](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Variable%20Frequency.jpeg)

After these analyses, it was concluded that some of the data will be dropped based on *materiality (will the dropped values significantly affect the analysis?)*, and the other will be encoded with integers. The *unknown* valuables were to be distributed back to the dataset in the same proportion the rest of the data was distributed, minus the *unknown* values.

Subsequently, unavailable and unknown datapoints were re-disrubuted within the dataset in the same proprtions as the known values in order to minimize the loss of data. Data that could not be salvaged was dropped, and the variables were encoded in integer forms; for example 0, 1, 2, and one unique identifier was retained for the dataset.

A new dataset by the name of "feature_df" was formulated after all the changes were made andd relevant predictor variables were chosen through which a machine learning model would be created.

Next, the data was split into a *training set* and a *testing set* in order to train our model and test the predictions it makes in order to get accuracy metrics.

Most importantly, the number of accidents that were *property damage only* and the number of accidents including *physical injury* were compared in order to check the balance of the data so that biases could be minimized.

#### Balanced or Unbalanced?

![Severity](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Severity%20of%20Accidents.jpeg)

After recognizing that the dataset was clearly imbalanced, a Python library called *Imbalanced-Learn* was imported and *SMOTE* was used to *balance* the data to reduce the possibility of inaccurate predictions caused by having a significantly higher number of *Property Damage Only* datapoints within our *training set*. If this step was omitted, the model would have predicted a lot more *0s* or *Property Damage Onlys* than it should have.

### Exploratory Data Analysis & Inferential Statistics

As a starting point, it was decided that any variable that is ~10% of the highest frequency variable that might cause an accident be included within the machine learning model and all 6 relevant variables fit this criterion. In order to check this, a bar graph was created and the frequencies were checked.

#### The Predictor Variables

![Accident Causes](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Accident%20Causes.jpeg)

Because the data was in a *cleaner* state now, it was easy to re-confirm that the *predictor variables* we had chosen by intuition were relevant whilst making the prediction and this was confirmed by the above visualization - most accidents had adverse conditions with respect to all the chosen variables.

### Model Selection

Subsequent to gaining a complete understanding of the dataset, it was evident that there were no *continuous* variables and hence, a classification model was to be used instead of a *regression model*. There were four options that could have been implemented - *Decision Tree, K-Nearest Neighbor, Logistic Regression, and Support Vector Machine.*

Because SVM's training complexity is largely reliant on the size of the dataset and is not well suited to larger datasets, it was not used in this specific case.

In the end, *Decision Tree, Logistic Regression, and KNN* models were shortlisted as the machine learning classification algroithms that were to be tested.

## Results

The results of each of the three models varied; one excelled at predicting the *positives* accurately while the other predicted the *negativves* better. It was evident by the results that the predictions could have been improved if there was a more complete dataset at hand.

### K-Nearest Neighbor

Before creating the KNN model, a loop was ran from range 1 to 10 where the accuracy of the model was checked with varying values of K, and *K = 4* was chosen as it produced the highest accuracy. The result can be seen below:

#### Choosing the right *K*

![K-Nearest Neighbor best K](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/K-Nearest%20Neighbor%20(KNN)%20best%20K%20Value.jpeg)

#### KNN Classification Metrics Report

![K-Nearest Neighbor](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/KNN%20Classification%20Metrics%20Report.JPG)

### Decision Tree

#### Decision Tree Classification Metrics Report

![Decision Tree](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Decision%20Tree%20Classification%20Metrics%20Report.JPG)

### Logistic Regression

The Logistic Regression model tends to falter with larger datasets containing a high frequency of minority datapoints unless a more complex, penalty-oriented model is used. Surprisngly, it was also able to make fair predictions relative to the other two models.

#### Logistic Regression Classification Metrics Report

![Logistic Regression](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Logistic%20Regression%20Classification%20Metrics%20Report.JPG)

It was noticed that the logistic regression model produced a higher-than-desried *uncertainty* which is illustrated by its *log loss*.

#### Log Loss

![Logistic Regression Log Loss](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Logistic%20Regression%20(Log%20Loss).JPG)

## Discussion

The results of all three machine learning models that were used varied significantly. One excelled at predicting the occurences of *0* while the other would predict *0* and *1* with ~50-50 accuracies.

#### Comparison of Accuracy Metrics across the Models

![Comparison of Accuracy Metrics](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Comparison%20of%20Accuracy%20Metrics.PNG)

In order to understand what the report above means, it is necessary to take a look at what precision and recall signify.

#### Precision VS Recall

![Precision VS Recall](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Precision%20VS%20Recall.png)



## Conclusion


## Sources

- https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(19)30170-6/fulltext#articleInformation
- By Walber - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=36926283

