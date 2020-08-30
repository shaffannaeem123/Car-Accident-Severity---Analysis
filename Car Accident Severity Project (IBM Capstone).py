#Importing initial packages:

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pip
import scipy
from sklearn import preprocessing
import sm as sm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import folium
import webbrowser
from folium import plugins
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

#Reading the Data:
import sm as sm

main_df = pd.read_csv ("C:\\Users\\Shaffan\\Desktop\\Car_Severity_Data.csv")

main_df.dtypes

#Encoding Accident Severity to 0 & 1 from 1 & 2:

severity_code = main_df ['SEVERITYCODE'].values

from sklearn import preprocessing

labels = preprocessing.LabelEncoder()

labels.fit([1, 2])

severity_code = labels.transform (severity_code)

#Adding the revised Severity Code Column back to the main_df:

main_df ["SEVERITYCODE"] = severity_code

#Checking whether the column has been succesfully added or not:

main_df ["SEVERITYCODE"]

descriptive_stats = main_df.describe (include = 'all')

descriptive_stats

#Changing INATTENTIONIND (Y to 1, 0 to N):

main_df ["INATTENTIONIND"].replace ("Y", 1, inplace = True)
main_df ["INATTENTIONIND"].replace (np.nan, 0, inplace = True)

#Checking if the conversion was accurate (1 should = 29, 805):

main_df ["INATTENTIONIND"].value_counts()

#Convert INATTENTIONIND to int:

main_df ["INATTENTIONIND"] = main_df ["INATTENTIONIND"].astype(int)

#Changing UNDERINFL (Y to 1, 0 to N):
main_df["UNDERINFL"].replace("N", 0, inplace=True)
main_df["UNDERINFL"].replace("Y", 1, inplace=True)
main_df ["UNDERINFL"] = main_df["UNDERINFL"]

#Changing SPEEDING (Y to 1, 0 to N):
main_df["SPEEDING"].replace(np.nan, 0, inplace=True)
main_df["SPEEDING"].replace("Y", 1, inplace=True)
main_df ["SPEEDING"] = main_df["SPEEDING"].astype('int')

#Check unique variables for further dummy conversions:

print ("WEATHER: ", main_df ["WEATHER"].unique())
print ("ROAD CONDITION", main_df ["ROADCOND"].unique())
print ("LIGHT CONDITION: ", main_df ["LIGHTCOND"].unique())
print ("SPEEDING OR NOT: ", main_df ["SPEEDING"].unique())
print ("UNDER THE INFLUENCE OR NOT: ", main_df ["UNDERINFL"].unique())
print ("INATTENTIVE OR NOT: ", main_df ["INATTENTIONIND"].unique())


#Checking if INCKEY can be used as unique identifier:

print ("Number of unique values: ", main_df ["INCKEY"].nunique())

#Changing LIGHTCOND to 0, 1, 2 (Dark, Medium, Daylight):

lightcond_counts = main_df["LIGHTCOND"].value_counts()

main_df["LIGHTCOND"].replace("Daylight", 0, inplace=True)
main_df["LIGHTCOND"].replace("Dark - Street Lights On", 1, inplace=True)
main_df["LIGHTCOND"].replace("Dark - No Street Lights", 2, inplace=True)
main_df["LIGHTCOND"].replace("Dusk", 1, inplace=True)
main_df["LIGHTCOND"].replace("Dawn", 1, inplace=True)
main_df["LIGHTCOND"].replace("Dark - Street Lights Off", 2, inplace=True)
main_df["LIGHTCOND"].replace("Dark - Unknown Lighting", 2, inplace=True)
main_df["LIGHTCOND"].replace("Other", "Unknown", inplace=True)

main_df["LIGHTCOND"]=main_df["LIGHTCOND"]

nan_values = main_df ["LIGHTCOND"].isna().sum()

sum_to_assign = lightcond_counts ["Unknown"]

print ("To assign: " , sum_to_assign)
print ("To Drop: ", nan_values)

main_df["LIGHTCOND"].value_counts()

light_length = main_df["LIGHTCOND"].size

import random

#Encoding Weather Conditions(0 = Clear, 1 = Overcast and Cloudy, 2 = Windy, 3 = Rain and Snow

main_df["WEATHER"].replace("Clear", 0, inplace=True)
main_df["WEATHER"].replace("Raining", 3, inplace=True)
main_df["WEATHER"].replace("Overcast", 1, inplace=True)
main_df["WEATHER"].replace("Other", "Unknown", inplace=True)
main_df["WEATHER"].replace("Snowing", 3, inplace=True)
main_df["WEATHER"].replace("Fog/Smog/Smoke", 2, inplace=True)
main_df["WEATHER"].replace("Sleet/Hail/Freezing Rain", 3, inplace=True)
main_df["WEATHER"].replace("Blowing Sand/Dirt", 2, inplace=True)
main_df["WEATHER"].replace("Severe Crosswind", 2, inplace=True)
main_df["WEATHER"].replace("Partly Cloudy", 1, inplace=True)

#Encoding Road Conditions(0 = Dry, 1 = Mushy, 2 = Wet)
main_df["ROADCOND"].replace("Dry", 0, inplace=True)
main_df["ROADCOND"].replace("Wet", 2, inplace=True)
main_df["ROADCOND"].replace("Ice", 2, inplace=True)
main_df["ROADCOND"].replace("Snow/Slush", 1, inplace=True)
main_df["ROADCOND"].replace("Other", "Unknown", inplace=True)
main_df["ROADCOND"].replace("Standing Water", 2, inplace=True)
main_df["ROADCOND"].replace("Sand/Mud/Dirt", 1, inplace=True)
main_df["ROADCOND"].replace("Oil", 2, inplace=True)

#Creating a new Pandas Dataframe with only Selected Features, Target Variabble, and Unique Identifier:

selected_columns = main_df[["INCKEY","INATTENTIONIND","UNDERINFL","WEATHER","ROADCOND","LIGHTCOND","SPEEDING","SEVERITYCODE"]]
feature_df = selected_columns.copy()
feature_df.dropna(axis = 0,how = 'any',inplace = True)
feature_df.reset_index(inplace = True)

#Distributing "Unknown" in LIGHTCOND:

lightcondsize = feature_df ["LIGHTCOND"].size


featureinlightcond = feature_df ['LIGHTCOND'] == 'Unknown'
featureinlightcond

lightcond = feature_df['LIGHTCOND']
lightcond = lightcond.values
lightcond = lightcond[featureinlightcond]

daylightbool = np.random.rand(len(lightcond))<=0.66
mediumlightbool = np.random.rand(len(lightcond))<=0.32
darklightbool = np.random.rand(len(lightcond))<=0.02

lightcondnzero = np.count_nonzero(lightcond)

lightcond[0:9036]=0
lightcond[9036:13417]=1
lightcond[13417:13691]=2

number = 12993

#Mapping the Unknowns back to feature_df in the same proportion:

feature_df.loc [feature_df.LIGHTCOND == "Unknown", 'LIGHTCOND'] = lightcond

roadcondsize = feature_df ["ROADCOND"].size

featureinroadcond = feature_df ['ROADCOND'] == 'Unknown'

roadcond = feature_df['LIGHTCOND']
roadcond = roadcond.values
roadcond = roadcond[featureinroadcond]

roadcond[0:9954]=0
roadcond[9954:10040]=1
roadcond[10040:15163]=2

feature_df.loc[feature_df.ROADCOND == "Unknown", 'ROADCOND'] = roadcond


weathersize = feature_df ["WEATHER"].size

featureinweather = feature_df ['WEATHER'] == 'Unknown'

weather = feature_df['WEATHER']
weather = weather.values
weather = weather[featureinweather]

weather[0:10151]=0
weather[10151:12683]=1
weather[12683:12742]=2
weather[12742:15864]=3

feature_df.loc[feature_df.WEATHER == "Unknown", 'WEATHER'] = weather
feature_df["WEATHER"]=feature_df["WEATHER"].astype(int)
feature_df["SPEEDING"]=feature_df["SPEEDING"].astype(int)
feature_df["INATTENTIONIND"]=feature_df["INATTENTIONIND"].astype(int)
feature_df ["ROADCOND"] = feature_df["ROADCOND"].astype(int)
feature_df ["LIGHTCOND"] = feature_df["LIGHTCOND"].astype(int)
feature_df ["UNDERINFL"] = feature_df["UNDERINFL"].astype(int)


feature_summary = feature_df.describe (include = 'all')

#Decision Tree Model

#Create Featureset Arrays

X = feature_df [['UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'SPEEDING']].values

y = feature_df ['SEVERITYCODE'].values

#Create test_train split:

from sklearn.model_selection import train_test_split

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=0)

#Check Shapes:

print (X_trainset.shape)
print (y_trainset.shape)

print (X_testset.shape)
print (y_testset.shape)

# Balance the Data

from imblearn.over_sampling import SMOTE
os = SMOTE (random_state=0)

os_data_X, os_data_y= os.fit_sample(X_trainset, y_trainset)

#Create the Decision Tree Model:

from sklearn.tree import DecisionTreeClassifier

AccidentTree = DecisionTreeClassifier(criterion = "entropy", max_depth = 6)
print (AccidentTree)

AccidentTree.fit(os_data_X, os_data_y)

#Make Prediction:

predTree = AccidentTree.predict(X_testset)

print (predTree [0:10])
print (y_testset [0:10])

shizer = pd.DataFrame (predTree)
shizer[0] = shizer [0].astype(int)

np.count_nonzero(shizer[0] == 0)
np.count_nonzero (y_testset)


#Check Accuracy:

from sklearn import metrics

print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
print (" Decision Tree's Jaccard Score: ", metrics.jaccard_score (predTree, y_testset))
print ("Decision Tree's F1 Score: ", f1_score (predTree, y_testset))
print ("Decision Tree's Classification Report: ", classification_report (predTree, y_testset))

DecisionTree_classification_report = classification_report (predTree, y_testset)

#KNN Model

#Import KNC

from sklearn.neighbors import KNeighborsClassifier as KNC

#Find Best K for our Model

Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfusionMx = [];
for n in range (1, Ks):
    # Train Model and Predict
    neigh = KNC(n_neighbors=n).fit(os_data_X, os_data_y)
    yhat = neigh.predict(X_testset)
    mean_acc[n - 1] = metrics.accuracy_score(y_testset, yhat)
    std_acc[n - 1] = np.std(yhat == y_testset) / np.sqrt(yhat.shape[0])



plt.plot (range(1,Ks),mean_acc,'g')
plt.fill_between (range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.05)
plt.legend (('Accuracy ', '+/- 3xstd'))
plt.ylabel ('Accuracy ')
plt.xlabel ('Number of Neighbors (K)')
plt.tight_layout ()
plt.show ()

print (mean_acc)

k = 4

neigh = KNC (n_neighbors = k).fit(os_data_X,os_data_y)

yhat = neigh.predict(X_testset)

from sklearn import metrics

from sklearn.metrics import classification_report

KNN_Classification_Report = classification_report (yhat, y_testset)

print("KNN Train set Accuracy: ", metrics.accuracy_score(y_trainset, neigh.predict(X_trainset)))
print("KNN Test set Accuracy: ", metrics.accuracy_score(y_testset, yhat))
print (f1_score(yhat, y_testset))
print (jaccard_score(yhat, y_testset))

#Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='newton-cg').fit(os_data_X, os_data_y)
LR

yhat = LR.predict(X_testset)
yhat

yhat_prob = LR.predict_proba(X_testset)
yhat_prob

from sklearn.metrics import jaccard_score

print("Logistic Regression Similarity Score: ", metrics.accuracy_score (yhat, y_testset))

print ("Logistic Regression Jaccard Score: ", jaccard_score(y_testset, yhat))

print ("Logistic Regression F1 Score: ", f1_score(y_testset, yhat))

from sklearn.metrics import log_loss
print ("Log Loss: ", log_loss(y_testset, yhat_prob))

from sklearn.metrics import classification_report

print ("Logistic Regression Classification Report: ", classification_report (yhat,y_testset))

logistic_classification_report = classification_report (yhat,y_testset)

# For Physical Injury (1)

prob_df = pd.DataFrame (yhat_prob)
prob_df_fifty_above = (prob_df [1] > 0.5)
prob_df_count = np.count_nonzero (prob_df)
fifty_above_count = np.count_nonzero (prob_df_fifty_above)

percentage_fifty_above = (fifty_above_count/prob_df_count)

np.count_nonzero (y_testset)
np.count_nonzero (yhat)

print ("Percentage of probabilities that are > 50% predicting Physical Injury (1)", percentage_fifty_above*100, "%")



#Visualizations

severity_v_count = main_df ["SEVERITYCODE"].value_counts()

severity_v_count.plot (kind = 'bar', figsize=(10, 8), rot=90,  alpha = 0.8, color = ['goldenrod', 'firebrick'])
plt.xlabel ("Property Damage (0) VS Physical Injury (1)", color = 'black', fontsize = 14)
plt.ylabel ("Frequency", color = 'black', fontsize = 14, labelpad= 20)
plt.xticks(rotation = 360)
plt.title ("Type of Accidents - Seattle, Washington", color = 'black', fontsize = 18, weight = 'bold')
plt.set_cmap('Accent_r')

colors = {"Property Damage Only":'goldenrod', "Physical Injury":'firebrick'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]


plt.legend(handles, labels)

plt.tight_layout()
plt.show()

#Latitude Area Plot with ROADCOND

area_plot_df = main_df [["Y", "ROADCOND"]].copy()

area_plot_df.sort_values (["Y"], ascending = 'true', inplace=True)

area_plot_df.dropna (axis = 0, how = 'any', inplace = True)

area_plot_df = area_plot_df [area_plot_df.ROADCOND != "Unknown"]

area_plot_df = area_plot_df.iloc [0:100000:1000, 0:]

area_plot_df = area_plot_df.transpose()

area_plot_df.plot(kind='area', alpha=0.25, stacked=False, figsize=(10, 8))

plt.show()

#Folium Map

import folium

world_map = folium.Map()
world_map

world_map.save('worldmap.html')

#Open Folium in Broswer HTML

import webbrowser

def auto_open(path):
    html_page = f'{path}'
    world_map.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)

auto_open('worldmap.html')

webbrowser.open('https://stackoverflow.com/questions/22445217/python-webbrowser-open-to-open-chrome-browser')


#Identifying the midpoint lat-long of our Data

mid_point_long_lat = main_df [["X", "Y"]].copy()

print ("Latitude", main_df ["Y"].median())
print ("Longitude", main_df ["X"].median())

#Folium Map

seattle_map = folium.Map(location=[47.61536892, -122.3302243], zoom_start=6)

#Reduce Computational Cost

limit = 100005
reduced_df = main_df.iloc [0:limit:5, 0:]
reduced_df.dropna (axis = 0, how = 'any', inplace = True)

from folium import plugins

# let's start again with a clean copy of the map of San Francisco
seattle_map = folium.Map(location=[47.61536892, -122.3302243], zoom_start=6)

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(seattle_map)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(reduced_df.Y, reduced_df.X, reduced_df.SEVERITYCODE):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

seattle_map.add_child(incidents)

# display map
seattle_map

#Display the map

seattle_map.save("seattlemap.html")

auto_open("seattlemap.html")

reduced_df.to_csv("reduced.html")

webbrowser.open("seattlemap.html")

webbrowser.open ("reduced.html")

#Further Visualizations

#Create new variables

light_viz = np.count_nonzero(feature_df ["LIGHTCOND"])
road_viz = np.count_nonzero(feature_df ["ROADCOND"])
weather_viz = np.count_nonzero(feature_df ["WEATHER"])
speeding_viz = np.count_nonzero(feature_df ["SPEEDING"])
underinfl_viz = np.count_nonzero(feature_df ["UNDERINFL"])
inattentive_viz = np.count_nonzero(feature_df ["INATTENTIONIND"])

#Create Variable Count Array

viz_count = [["Adverse Lighting Conditions", "Adverse Road Conditions", "Adverse Weather Conditions", "Over-speeding", "Under the Influence", "Inattentive Driving" ],
             [light_viz, road_viz, weather_viz, speeding_viz, underinfl_viz, inattentive_viz]]
viz_count

#Create a new DataFrame with new count variables

viz_df_count = pd.DataFrame (viz_count)
viz_df_count = viz_df_count.transpose()
viz_df_count.set_index(0, inplace=True)

#Bar Graph

colors_2 = ['firebrick','indianred', 'darkred', 'darksalmon',  'salmon', 'tomato']

bar_viz = viz_df_count.plot (kind = 'barh', color = [colors_2], figsize=(10, 8), alpha = 0.75 )
plt.xlabel ("Frequency", fontsize = 14, labelpad = 20)
plt.ylabel (None)
plt.title ("Accident Causes - Seattle, Washington", fontsize = 16, weight = 'bold')
plt.tight_layout()

colors = {'Adverse Lighting Conditions':'firebrick', 'Adverse Road Conditions':'indianred', 'Adverse Weather Conditions':'darkred',
          'Over-speeding':'darksalmon', 'Under the Influence':'salmon', 'Inattentive Driving':'tomato'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]


plt.legend(handles, labels)
plt.show()

import matplotlib

np.count_nonzero (feature_df["UNDERINFL"])





