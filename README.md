# Car Accident Severity Analysis: Seattle, Washington (Capstone Project)

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

In this project, I aim to predict whether an accident that happens under a specific set of circumstances will be an accident limited to *property damage* or if it will include some form of *physical injury* to the driver and passengers.

## Data

The dataset that I am using for this project is majorly provided by the government and pertains to the city of Seattle, Washington. It is recent data and includes observations from 2004 to 2020. The number of observations (194, 6773) in the data is enough to formulate a machine learning model. A large majority of the feature-set contains qualitative and categorical data, which is why performing a simple *Multiple Linear Regression* or* Polynomial Regression* is not the best option. The target variable for this model would be the level of severity of the car accident (property damage only versus physical injury).
After initial data exploration, we determined the following features to be most relevant when predicting the *Accident Severity*.
This is the dataset in CSV format in case you want to view it: (https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv)

### Independent Variables

- INATTENTIONIND: Whether or not collision was due to inattention
- UNDERINFL: Whether or not a driver involved was under the influence of drugs or alcohol
- WEATHER: The weather conditions during the time of the collision
- ROADCOND: The condition of the road during the collision
- LIGHTCOND: The light conditions during the collision
- SPEEDING: Whether or not speeding was a factor in the collision

### Target Variable

- SEVERITYCODE: A code that corresponds to the severity of the collision

## Methodology

### Data Collection

The dataset used for this project is a public dataset and illustrates the circumstances in which car accidents take place in Seattle, Washington from 2004 to 2020.

### Data Cleaning

After gaining an understanding of the problem, I moved on to understanding the data that I had to figure out how to transform it to its most meaningful form. The first thing I did was check the data types of each variable and then I explored how many variables were missing some values.

The data types in the Seattle dataset mostly comprised *categorical variables* and *objects*; I instantly concluded that a Simple/Multiple/Polynomial Regression would not work here. I held that thought and confirmed it later. The variables were noted down in plain english and most of them were to be encoded with integers in an *ordinal* manner. The data types of the original dataset can be viewed below:

![Data Types](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Data%20Types.JPG)

The frequency of meaningful data points within each variable column can be seen below:

![Data Exploration](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Variable%20Frequency.jpeg)

After these analyses, it was concluded that some of the data will be dropped based on *materiality*, and the other will be encoded with integers. The *unknown* valuables were to be distributed back to the dataset in the same proportion the rest of the data was distributed.

In the end, unavailable and unknown datapoints were re-disrubuted within the dataset in the same proprtions as the known values in order to minimize the loss of data. Data that could not be salvaged was dropped, and the variables were encoded in integer forms; for example 0, 1, 2 and one unique identifier was retained for the dataset.

A new dataset by the name of "feature_df" was formulated after all the changes were made andd relevant predictor variables were chosen through which a machine learning model would be created.

### Exploratory Data Analysis & Inferential Statistics

As a starting point, I decided that any variable that is ~10% of the highest frequency variable that might cause an accident be included within the machine learning model and all 6 relevant variables fit this criterion. In order to check this, a bar graph was created and the frequencies were checked.

![Accident Causes](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Accident%20Causes.jpeg)

Second, the number of accidents that were *property damage only* and the number of accidents including *physical injury* were separated and the frequencies were compared in order to check the balance of the data.

![Severity](https://github.com/shaffannaeem123/Car-Accident-Severity---Analysis/blob/master/Severity%20of%20Accidents.jpeg)

### Model Selection





## Sources

- https://www.thelancet.com/journals/lanplh/article/PIIS2542-5196(19)30170-6/fulltext#articleInformation

