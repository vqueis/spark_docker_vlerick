# # Data Exploration 

# %%
import numpy as np
np.warnings.filterwarnings('ignore')

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# %%
dir = '../data/pre_release.csv'
df_pre = pd.read_csv(dir)
df_pre.head(10)

# %%
dir = '../data/after_release.csv'
df_after = pd.read_csv(dir)
df_after.head(10)

# %%


# %% [markdown]
# #### Storyline
# After looking at the pre-release and after-release data, I am very curious if I would be able to predict the movie_facebook_likes. I expect that this depends on various factors, such as the actors facebook likes, but also the directors facebook likes, cast fb likes, the genre of the movie and the budget that the mvoie had. So my target variable will be to predict the movie_facebook_likes (from the after-release dataset)
# 
# 
# <center><img src="https://i.pinimg.com/originals/39/44/6c/39446caa52f53369b92bc97253d2b2f1.png" width="200" style="float:right"></center>

# %% [markdown]
# 

# %%
#now we merge the pre and after release
df_merged = pd.merge(df_pre, df_after, on='movie_title', how='inner')
df_merged


# %%
df_merged.shape #we have 1069 records and 22 columns

# %%
df_merged.info()


# %% [markdown]
# ### Variables of interest

# %% [markdown]
# 
# 1. director_facebook_likes 
# 2. actor_1_facebook_likes 
# 3. actor_2_facebook_likes 
# 4. actor_3_facebook_likes 
# 5. cast_total_facebook_likes
# 6. Budget
# 7. Genres
# 8. Country
# 
# I think that the listed variables above can be very interesting for further analysis.
# For example, I think that the more popular the director is and the actors (cast members) (measured by their fb likes), the more popular the movie also will be on facebook (movie_facebook_likes)
# Additionally, when the budget is higher, I also expect the movie to have more facebook likes because more budget means probably also more marketing budget en thus potentially more likes
# Lastly, genres and countries can also influence the movie_facebook_likes I suspect. Certain movie genres are just more popular among people, and there are just certain countries that produce more popular movies in general...

# %%
df_merged.describe().round(2) #let's see some stats
#apperently for actor 3, 1 and 2 there are some missings because they do not add up to 1069

# %%
df_merged.describe(include='object') 
#We see that USA is the most prevalent country, 780 times in our dataset and English is the most common 
#language of the movies


# %%
categoricals = df_merged.select_dtypes(include = 'object').columns.tolist()
# print unique values

categoricals  #list of all categories in our dataset and their unique values

for c in categoricals:
    print(c, "has values: ", df_merged[c].unique()) 

# %%
df_merged['genres'].value_counts() #here we see that genres still has the |, we have to solve this later on

# %%
df_merged['actor_1_name'].value_counts()

# %%
df_merged['actor_2_name'].value_counts() 

# %%
df_merged['actor_3_name'].value_counts()

# %%
df_likes1 = df_merged[['actor_1_name', 'actor_1_facebook_likes']] #with this new dataframe I can see which actor 
#has the most facebook likes
df_likes.sort_values('actor_1_facebook_likes', ascending=False)
#apparently max amount of likes is 1000 on Facebook, which is not very realistic. 
#We also see some NaN appearing, which I will deal with later

# %%
df_likes2 = df_merged[['actor_2_name', 'actor_2_facebook_likes']]
df_likes2.sort_values('actor_2_facebook_likes', ascending=False)

# %%
df_likes3 = df_merged[['actor_3_name', 'actor_3_facebook_likes']]
df_likes3.sort_values('actor_3_facebook_likes', ascending=False)

# %%
df_likes4 = df_merged[['director_name', 'director_facebook_likes']]
df_likes4.sort_values('director_facebook_likes', ascending=False)
#A director gets way less likes on facebook than the actors, sadly enough. 

# %%
df_merged['country'].value_counts()

# %%
#We have a lot of countries and the majority of the coutries produces appear 1 or 2 movies in the dataset. 
#This makes the dataset and the variables in the models later on unnecessarily big, therefore I will now only
#select those countries that occur more than 10 times in the dataset
country_df = df_merged['country'].value_counts().to_frame()
countries_other = set()
countries_main = set()
for i, row in country_df.iterrows():
    if row['country'] < 10:
        countries_other.add(i)
    else:
        countries_main.add(i)
        
#Now we replace all these names in the original dataset with 'Other'
for c in countries_other:
    df_merged['country'].replace(c, 'Other_country', inplace = True)
    
#Add 'Other' to the set of unique countries
countries_main.add('Other_country')

# Create a new column for each unique value
for c in countries_main:
    df_merged[c] = 0
    df_merged[c] = df_merged[c].astype('int')

    # Set the corresponding columns to 1
for i, row in df_merged.iterrows():
    df_merged.loc[i, row['country']] = 1

# %%
df_merged.head()
#We see that we are now left with the following countries:
#Germany, UK, Australia, USA, Canada and France. All other countries are grouped under "Other_Countries"


# %% [markdown]
# #  Data Cleaning

# %%
df_merged.isnull() 

# %%
df_merged.isnull().sum() # gives most missing values back, we can see that sex has most missings
df_merged[df_merged.isnull().any(axis=1)]# check row
df_merged.isnull().any(axis=1) #gives per row back if there are missings (if missing, TRUE)




# %%
df_merged = df_merged[df_merged.isnull().sum(axis=1) < 5] #we have deleted some rows, but we do not know how manu we deleted exactly
df_merged.shape 
df_merged.isnull().sum() #there are still some missing values left
#we have to look more into
#actor_3_facebook_likes
#actor_3_name      
#language 
#content_rating 
#num_critic_for_reviews 

# %%
print(df_merged['actor_3_facebook_likes'].
      value_counts(dropna = False)) #problem is, python only recognises the NaN as missing value. 

# %%
print(df_merged['movie_facebook_likes'].
      value_counts(dropna = False)) 

#this variable has a lot of 0 values

# %%
cat = df_merged.select_dtypes(include = 'object').columns.tolist() #select all categorical column names and store them in a list
num = df_merged.select_dtypes(include = 'float64').columns.tolist() #select all numerical column names and store them in a list

# %%
cat_imputer = SimpleImputer(strategy = "most_frequent") 
cat_imputer.fit(df_merged[cat])
df_merged[cat] = cat_imputer.transform(df_merged[cat])
df_merged[cat].isnull().sum()






# %%
num_imputer = SimpleImputer(strategy='median') #just the strategy changes for numerical data
df_merged[num] = num_imputer.fit_transform(df_merged[num])



# %%
for column in cat:
    df_merged[column].fillna(df_merged[column].mode(dropna = True), inplace = True) 
for column in num: 
    df_merged[column].fillna(df_merged[column].median(), inplace = True) 

# %%
df_merged.isnull().sum() #data cleaning has worked, every variable has now 0 missings

# %%
df_merged[df_merged.duplicated() == True] #check for duplicates

# %%
df_merged.drop_duplicates(inplace=True)
print(df_merged[df_merged.duplicated()== True].shape[0])
#remove duplicates

# %%
print(df_merged[df_merged.duplicated() == True].shape[0]) #there are no duplicates anymore

# %% [markdown]
# ### Outliers 

# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(data=
            df_merged[['actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes']]); 

#We do see here in this boxplot that the cast total facebook likes have some outliers

# %%
sns.set(rc={'figure.figsize':(15,12.5)})
sns.boxplot(data=
            df_merged[['actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes']]); 

sns.stripplot(data=
            df_merged[['actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes']]); 

#here I made a jitter plot to plot all points on the boxplot and see the spread
#again we can see there are quite some outliers int the cast total facebook likes, however,
#as these likes of the cast can also have an influence on the movie_facebook_likes (popular cast --> potentially more facebook likes)
#I won't delete the outliers

# %%
import seaborn as sns
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

sns.histplot(data=df_merged, x="director_facebook_likes", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=df_merged, x="actor_1_facebook_likes", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df_merged, x="actor_2_facebook_likes", kde=True, color="gold", ax=axs[1, 0])
sns.histplot(data=df_merged, x="actor_3_facebook_likes", kde=True, color="teal", ax=axs[1, 1])

plt.show()

#actor 1 is quite popular, then actor 2 and actor 3 is least popular at least in terms of fb_likes...

# %% [markdown]
# ## Correlations

# %%
df_merged.corr()

# %%
sns.heatmap(df_merged.corr());

#we do see some potential problems in terms of multicollinearity
#because actor facebook likes are relatively highly correlated with one another
#this brings questions for the model that I want to create... which variables to put in the regression and which ones not
#actor 3 fb likes is for example positivily correlated with actor 1 fb likes and cast fb likes (which is MC threat)
#actor 1 also very highly correlated with cast fb likes


# %%
sns.scatterplot(data=df_merged, x="actor_1_facebook_likes", y="cast_total_facebook_likes");
                

# %%
sns.scatterplot(data=df_merged, x="actor_2_facebook_likes", y="cast_total_facebook_likes");
                

# %%
sns.scatterplot(data=df_merged, x="actor_3_facebook_likes", y="cast_total_facebook_likes");
                

# %% [markdown]
# #### analysis of scatter plot and correlations
# based on these scatter plots and the correlation matrix we can conclude that there is a strong positive relationship between actor fb likes and the cast fb likes
# 

# %%


# %% [markdown]
# ### Next steps
# 
# now that the data is cleaned and we looked at the potential issues such as outliers and MC, I think we are well prepped to start modelling. So I have two ideas. 
# 1. first I am going to check just a linear regression, and I will be stupid at first and put all my variables in it that I initially thought would be predicitve. I do know there is MC, but that is why in a second step I will apply LASSO to see which variables get dropped. Then I am going to check the validity and predicitve power of this model
# 
# 2. Next, I am going to make a random forrest and compare this with my first model and decide which one is better

# %% [markdown]
# # Regression Model
# still need to make some dummies for the categorical variables like country and genres
# <center><img src="https://cdn-icons-png.flaticon.com/512/7440/7440405.png" width="20%" align="right" style></center>

# %%
x = df_merged['genres'].str.get_dummies(sep = '|')#This code separates the genres in the same cell using the 
#delimiter '|' and then makes dummy variables for all genres
combined_frames = [df_merged, x]
combined_df = pd.concat(combined_frames, axis = 1) #Concatinating the dummy variables for the genres to our dataset
combined_df = combined_df.drop('genres', axis = 1)

# %%
combined_df.shape

# %%
combined_df.head()

# %%
combined_df.drop(columns = ["actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "movie_facebook_likes", "language"])

# %%
#this one will make dummies for the country variable
x = combined_df['country'].str.get_dummies(sep = " ")
combined_frames = [combined_df, x]
combined_df2 = pd.concat(combined_frames, axis = 1) 
combined_df2 = combined_df2.drop('country', axis = 1) 

# %%
combined_df2.drop(columns = ["actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "language"])

# %%
# Now i finally have my combined_df2 with dummies for country AND genres correctly...
#we can now start building models
#only detail that can be included is to only take those countries that appear more than 10 times, bc we have
# a lot of countries that do not really matter now

# %%
combined_df2.shape

# %%
#Define dependent and independent variables
x = combined_df.drop(columns = ["country", "actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "language", "movie_facebook_likes", "duration"]) #remove irrelvant independent variables & target variable
y = combined_df["movie_facebook_likes"] #extract target variable
print(x)
print(y)

#I did keep in budget and genres because this might have explanatory power still to predict popularity of the movie

# %%
#randomly split into training (70%) and val (30%) sample
from sklearn.model_selection import train_test_split
seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)

# %%
#run a regression model with statmodels 
import statsmodels.api as sm

# first  add intercept to X (since not automatically included in ols estimation):
xc_train = sm.add_constant(x_train)
xc_val = sm.add_constant(x_val)
#train model
mod = sm.OLS(y_train,xc_train)
olsm = mod.fit()
#output table with parameter estimates (in summary2)
olsm.summary2().tables[1][['Coef.','Std.Err.','t','P>|t|']]

#So imagine my p-value would be 10%, the following variables are significant:
#actor_2_facebook_likes	
#UK
#Australia
#France
#Comedy
#Horror
#War


# %%
#predict
#Make a predictions and ass this back to the dataframe called val_pred
array_pred = np.round(olsm.predict(xc_val),0) #adjust round depending on predictions

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred



# %%
#Evaluate model: R-square & MAE
#by comparing actual and predicted value 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
#input below actual and predicted value from dataframe
act_value = val_pred["movie_facebook_likes"]
pred_value = val_pred["y_pred"]
#run evaluation metrics
rsquare = r2_score(act_value, pred_value)
mae = mean_absolute_error(act_value, pred_value)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})

#it is not a very good model, with even a negative R-Square...



# %%
y_val = np.array(y_val)
pred_value = np.array(pred_value)
errors = abs(pred_value - y_val)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_val)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

#accuracy gives a very strange output, but this is to blame on the 0 values in y_pred which 
#causes the accuracy to not work (optimally). 

# %% [markdown]
# ### Model Validation 
# 
# 

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import KFold
import numpy as np
x = np.concatenate((x_train, x_val))
y = np.concatenate((y_train, y_val))
k=5
kf = KFold(n_splits=k,random_state=2022,shuffle=True)
kf.get_n_splits(x)
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42, 
                                   max_depth=9, min_samples_leaf=25) 
for train_index, val_index in kf.split(x):
    clf_model.fit(x_train,y_train)
    x_train, x_val = x[train_index,:], x[val_index,:]
    y_train, y_val = y[train_index], y[val_index]
    clf_model.fit(x_train,y_train)
    print('Test Accuracy:',accuracy_score(y_val, clf_model.predict(x_val)))

    
#The test accuracy is the accuracy of a model on examples it hasn't seen.
#the accuracy is not that high, which will result in predictions that are not that good





# %%
# Cross validation
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
clf_1 = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=9, min_samples_leaf=25)
clf_2 = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None, min_samples_leaf=1) #clf_2 is gonna overfit
scores1 = cross_val_score(clf_1, x, y, cv=10)[:, None] #cv stands for doing the cross fold 10 times
scores2 = cross_val_score(clf_2, x, y, cv=10)[:, None]
_ = plt.boxplot(np.concatenate((scores1, scores2), axis=1))
plt.show()

# %%
#In a way I already did based on the significance level a selection of my variables for a potential second model
#See cell 103 in comments
#But I also wanted to perform a LASSO / Ridge regression to see what the outcome would be

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
X = combined_df[['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
       'director_facebook_likes', 'cast_total_facebook_likes', 'budget']]
features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
       'director_facebook_likes', 'cast_total_facebook_likes', 'budget']
lr = LinearRegression().fit(X, (combined_df['movie_facebook_likes']))
lasso = Lasso(alpha=0.5).fit(X, (combined_df['movie_facebook_likes']))
ridge = Ridge(alpha=1).fit(X, (combined_df['movie_facebook_likes']))
pd.DataFrame({'feature': [f'X_{i}' for i in range(6)],
     'lr_coef': lr.coef_,
     'lasso_coef': lasso.coef_,
     'ridge_coef': ridge.coef_})

#so of these 6 variables that I expect to have biggest influence on movie_facebook_likes
#all the coefficients are very low which means that it is equivalent to a linear regression, no variables will be dropped
#This was not helpful for further analysis of my model

# %% [markdown]
# ### explanation ridge & lasso 
# 
# source = https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/
# <center><img src="https://thumbs.dreamstime.com/b/lasso-icon-icon-cartoon-style-isolated-vector-illustration-88303760.jpg" width="20%" align="right" style></center>
# 
# 1. Ridge Regression:
# Performs L2 regularization, i.e. adds penalty equivalent to square of the magnitude of coefficients
# Minimization objective = LS Obj + α * (sum of square of coefficients)
# It includes all (or none) of the features in the model. Thus, the major advantage of ridge regression is coefficient shrinkage and reducing model complexity.
# It is majorly used to prevent overfitting. Since it includes all the features, it is not very useful in case of exorbitantly high #features, say in millions, as it will pose computational challenges.
# 2. Lasso Regression:
# Performs L1 regularization, i.e. adds penalty equivalent to absolute value of the magnitude of coefficients
# Minimization objective = LS Obj + α * (sum of absolute value of coefficients)
# Along with shrinking coefficients, lasso performs feature selection as well. As we observed earlier, some of the coefficients become exactly zero, which is equivalent to the particular feature being excluded from the model.
# Since it provides sparse solutions, it is generally the model of choice (or some variant of this concept) for modelling cases where the #features are in millions or more. In such a case, getting a sparse solution is of great computational advantage as the features with zero coefficients can simply be ignored.
# 
# 
# 
# 
# 

# %% [markdown]
# ### Regression model evaluation 1
# All in all is this first attempt of a regression not ideal. It gives a negative R-square (but close to zero), 
# there are also still variables in this model that have multicollineairity 
# and the accuracy is low
# So, let's try another model to predict movie_facebook_likes
# 

# %% [markdown]
# # Regression Forest 1
# 

# %%
#Let's see what this gives in a regression forest
x = combined_df.drop(columns = ["country", "actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "language", "movie_facebook_likes", "duration"]) #remove irrelvant independent variables & target variable
y = combined_df["movie_facebook_likes"] #extract target variable
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)
print(x_train.shape)
print(x.columns.shape)

# %%
#run a regression tree
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(max_depth=1, min_samples_leaf =1, random_state=seed).fit(x_train, y_train)
#show feature importance
pd.DataFrame({'category': x_train.columns,'importance':rfreg.feature_importances_}).set_index('category').sort_values(by = 'importance', ascending = False)

#I am using the regressor and not classifier as my target variable is not labeled but just numerical output

#In the feature importance of the random forest we can see that actor 2, cast, actor 1, actor 3 
#and the director and comedy are important 

# %%
#predict regression forest
array_pred = np.round(rfreg.predict(x_val),0)

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred

# %%
#Evaluate model: R-square & MAE
#by comparing actual and predicted value 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
#input below actual and predicted value from dataframe
act_value = val_pred["movie_facebook_likes"]
pred_value = val_pred["y_pred"]
#run evaluation metrics
rsquare = r2_score(act_value, pred_value)
mae = mean_absolute_error(act_value, pred_value)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})

# %%
y_val = np.array(y_val)
pred_value = np.array(pred_value)
errors = abs(pred_value - y_val)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_val)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# %% [markdown]
# # Regression model 2

# %% [markdown]
# ##### clearly, the regression model above is not working out, it even gives a negative r square which means that the model sucks balls. Now I will only include those variables that were significant in the previous model to check if there is any improvement.
# 

# %%
#Define dependent and independent variables
combined_df_model2 = combined_df.drop(columns = ["director_facebook_likes", "actor_3_facebook_likes", "actor_1_facebook_likes", "movie_facebook_likes", "director_name", "duration", "actor_1_name", 
                              "actor_2_name", "actor_3_name", "movie_title", "language", "content_rating", "cast_total_facebook_likes", "budget",
                             "num_critic_for_reviews", "gross", "num_voted_users", "imdb_score", "num_user_for_reviews", "country", "Other_country", "Music", "Musical",
                               "Mystery", "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "Western", "History", "Fantasy", "Crime", "Biography", "Family",
                               "Documentary", "Animation", "Adventure", "USA", "Germany", "Canada", "Action", "Drama"]) #remove irrelvant independent variables
x = combined_df_model2
y = combined_df["movie_facebook_likes"] #extract target variable
print(x)
print(y)

# %%
#randomly split into training (50%) and val (50%) sample
from sklearn.model_selection import train_test_split
seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.5, random_state = seed)

#even with 50-50 training and test

# %%
#run a regression model with statmodels (with sklearn no significance output)
import statsmodels.api as sm

# first  add intercept to X (since not automatically included in ols estimation):
xc_train = sm.add_constant(x_train)
xc_val = sm.add_constant(x_val)
#train model
mod = sm.OLS(y_train,xc_train)
olsm = mod.fit()
#output table with parameter estimates (in summary2)
olsm.summary2().tables[1][['Coef.','Std.Err.','t','P>|t|']]



# %%
#predict
#Make a predictions and ass this back to the dataframe called val_pred
array_pred = np.round(olsm.predict(xc_val),0) #adjust round depending on predictions

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred

# %%
#Evaluate model: R-square & MAE
#by comparing actual and predicted value 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
#input below actual and predicted value from dataframe
act_value = val_pred["movie_facebook_likes"]
pred_value = val_pred["y_pred"]
#run evaluation metrics
rsquare = r2_score(act_value, pred_value)
mae = mean_absolute_error(act_value, pred_value)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})

#did get a positive R-Square now, but it is almost 0, so the model still has no predicitive power. 

# %% [markdown]
# ### Model validation 
# 

# %%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

from sklearn.model_selection import KFold
import numpy as np
x = np.concatenate((x_train, x_val))
y = np.concatenate((y_train, y_val))
k=5
kf = KFold(n_splits=k,random_state=2022,shuffle=True)
kf.get_n_splits(x)
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42, 
                                   max_depth=9, min_samples_leaf=25) 
for train_index, val_index in kf.split(x):
    clf_model.fit(x_train,y_train)
    x_train, x_val = x[train_index,:], x[val_index,:]
    y_train, y_val = y[train_index], y[val_index]
    clf_model.fit(x_train,y_train)
    print('Test Accuracy:',accuracy_score(y_val, clf_model.predict(x_val)))

# %%
# Cross validation
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
clf_1 = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=9, min_samples_leaf=25)
clf_2 = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=None, min_samples_leaf=1) #clf_2 is gonna overfit
scores1 = cross_val_score(clf_1, x, y, cv=10)[:, None] #cv stands for doing the cross fold 10 times
scores2 = cross_val_score(clf_2, x, y, cv=10)[:, None]
_ = plt.boxplot(np.concatenate((scores1, scores2), axis=1))
plt.show()

# %% [markdown]
# ### regression evaluation 2 
# this second regression already gives a better R-squared, it became positive but it is very close to zero
# which basically means it is still not powerful to predict movie_facebook_likes
# 
# 
# 

# %% [markdown]
# # Regression forest 2 

# %%
combined_df_model2 = combined_df.drop(columns = ["director_facebook_likes", "actor_3_facebook_likes", "actor_1_facebook_likes", "movie_facebook_likes", "director_name", "duration", "actor_1_name", 
                              "actor_2_name", "actor_3_name", "movie_title", "language", "content_rating", "cast_total_facebook_likes", "budget",
                             "num_critic_for_reviews", "gross", "num_voted_users", "imdb_score", "num_user_for_reviews", "country", "Other_country", "Music", "Musical",
                               "Mystery", "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "Western", "History", "Fantasy", "Crime", "Biography", "Family",
                               "Documentary", "Animation", "Adventure", "USA", "Germany", "Canada", "Action", "Drama"]) #remove irrelvant independent variables
x = combined_df_model2
y = combined_df["movie_facebook_likes"] #extract target variable
print(x)
print(y)

# %%
#run a regression tree
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(max_depth=1, min_samples_leaf =1, random_state=seed).fit(x_train, y_train)
#show feature importance
pd.DataFrame({'category': x_train.columns,'importance':rfreg.feature_importances_}).set_index('category').sort_values(by = 'importance', ascending = False)



# %%
#predict regression forest
array_pred = np.round(rfreg.predict(x_val),0)

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred

# %%
#Evaluate model: R-square & MAE
#by comparing actual and predicted value 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
#input below actual and predicted value from dataframe
act_value = val_pred["movie_facebook_likes"]
pred_value = val_pred["y_pred"]
#run evaluation metrics
rsquare = r2_score(act_value, pred_value)
mae = mean_absolute_error(act_value, pred_value)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})




