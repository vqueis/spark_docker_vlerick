### 1. Read the CSV data from this S3 bucket using PySpark ###
from pyspark import SparkConf
from pyspark.sql import SparkSession

BUCKET = "dmacademy-course-assets"
KEY = "vlerick/after_release.csv", "vlerick/pre_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

df = spark.read.csv(f"s3a://{BUCKET}/{KEY}", header=True)
df.show()


# Print the schema of the DataFrame
df.printSchema()

### 2. Convert the Spark DataFrames to Pandas DataFrames ###
# Import the necessary modules
import pandas as pd

# Convert the Spark DataFrame to a Pandas DataFrame
pandas_df = df.toPandas()

# Print the first five rows of the Pandas DataFrame
print(pandas_df.head())

### 3. Rerun the same ML training and scoring logic ###
# ## that you had created prior to this class, starting with the Pandas DataFrames you got in step 2 ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
dir = '../data/pre_release.csv'
df_pre = pd.read_csv(dir)
df_pre.head(10)

dir = '../data/after_release.csv'
df_after = pd.read_csv(dir)
df_after.head(10)
#now we merge the pre and after release
df_merged = pd.merge(df_pre, df_after, on='movie_title', how='inner')
df_merged
df_merged.shape
df_merged.info
df_merged.describe().round(2)
df_merged.describe(include='object') 
categoricals = df_merged.select_dtypes(include = 'object').columns.tolist()
# print unique values

categoricals  #list of all categories in our dataset and their unique values

for c in categoricals:
    print(c, "has values: ", df_merged[c].unique()) 


df_merged['genres'].value_counts() 
df_merged['actor_1_name'].value_counts()
df_likes1 = df_merged[['actor_1_name', 'actor_1_facebook_likes']] #with this new dataframe I can see which actor 
#has the most facebook likes
df_likes.sort_values('actor_1_facebook_likes', ascending=False)
#apparently max amount of likes is 1000 on Facebook, which is not very realistic. 
#We also see some NaN appearing, which I will deal with later

df_likes2 = df_merged[['actor_2_name', 'actor_2_facebook_likes']]
df_likes2.sort_values('actor_2_facebook_likes', ascending=False)
df_likes3 = df_merged[['actor_3_name', 'actor_3_facebook_likes']]
df_likes3.sort_values('actor_3_facebook_likes', ascending=False)
df_likes4 = df_merged[['director_name', 'director_facebook_likes']]
df_likes4.sort_values('director_facebook_likes', ascending=False)
#A director gets way less likes on facebook than the actors, sadly enough. 

df_merged['country'].value_counts()
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

df_merged.isnull() 
df_merged.isnull().sum() # gives most missing values back, we can see that sex has most missings
df_merged[df_merged.isnull().any(axis=1)]# check row
df_merged.isnull().any(axis=1) #gives per row back if there are missings (if missing, TRUE)
df_merged = df_merged[df_merged.isnull().sum(axis=1) < 5] #we have deleted some rows, but we do not know how manu we deleted exactly
df_merged.shape 
df_merged.isnull().sum()
print(df_merged['actor_3_facebook_likes'].
      value_counts(dropna = False))
    
print(df_merged['movie_facebook_likes'].
      value_counts(dropna = False)) 

cat = df_merged.select_dtypes(include = 'object').columns.tolist() #select all categorical column names and store them in a list
num = df_merged.select_dtypes(include = 'float64').columns.tolist() #select all numerical column names and store them in a list

cat_imputer = SimpleImputer(strategy = "most_frequent") 
cat_imputer.fit(df_merged[cat])
df_merged[cat] = cat_imputer.transform(df_merged[cat])
df_merged[cat].isnull().sum()

num_imputer = SimpleImputer(strategy='median') #just the strategy changes for numerical data
df_merged[num] = num_imputer.fit_transform(df_merged[num])

for column in cat:
    df_merged[column].fillna(df_merged[column].mode(dropna = True), inplace = True) 
for column in num: 
    df_merged[column].fillna(df_merged[column].median(), inplace = True) 

df_merged.isnull().sum()

df_merged[df_merged.duplicated() == True] #check for duplicates

df_merged.drop_duplicates(inplace=True)
print(df_merged[df_merged.duplicated()== True].shape[0])
#remove duplicates

print(df_merged[df_merged.duplicated() == True].shape[0])

sns.set(rc={'figure.figsize':(15,12.5)})
sns.boxplot(data=
            df_merged[['actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes']]); 

sns.stripplot(data=
            df_merged[['actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes', 'cast_total_facebook_likes', 'director_facebook_likes']]); 

import seaborn as sns
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

sns.histplot(data=df_merged, x="director_facebook_likes", kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(data=df_merged, x="actor_1_facebook_likes", kde=True, color="olive", ax=axs[0, 1])
sns.histplot(data=df_merged, x="actor_2_facebook_likes", kde=True, color="gold", ax=axs[1, 0])
sns.histplot(data=df_merged, x="actor_3_facebook_likes", kde=True, color="teal", ax=axs[1, 1])

plt.show()

df_merged.corr

sns.scatterplot(data=df_merged, x="actor_1_facebook_likes", y="cast_total_facebook_likes");
sns.scatterplot(data=df_merged, x="actor_2_facebook_likes", y="cast_total_facebook_likes");
sns.scatterplot(data=df_merged, x="actor_3_facebook_likes", y="cast_total_facebook_likes");

x = df_merged['genres'].str.get_dummies(sep = '|')#This code separates the genres in the same cell using the 
#delimiter '|' and then makes dummy variables for all genres
combined_frames = [df_merged, x]
combined_df = pd.concat(combined_frames, axis = 1) #Concatinating the dummy variables for the genres to our dataset
combined_df = combined_df.drop('genres', axis = 1)

combined_df.shape
combined_df.head()
combined_df.drop(columns = ["actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "movie_facebook_likes", "language"])


#this one will make dummies for the country variable
x = combined_df['country'].str.get_dummies(sep = " ")
combined_frames = [combined_df, x]
combined_df2 = pd.concat(combined_frames, axis = 1) 
combined_df2 = combined_df2.drop('country', axis = 1) 

combined_df2.drop(columns = ["actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "language"])

#Define dependent and independent variables
x = combined_df.drop(columns = ["country", "actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "language", "movie_facebook_likes", "duration"]) #remove irrelvant independent variables & target variable
y = combined_df["movie_facebook_likes"] #extract target variable
print(x)
print(y)

#randomly split into training (70%) and val (30%) sample
from sklearn.model_selection import train_test_split
seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)

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

#Make a predictions and ass this back to the dataframe called val_pred
array_pred = np.round(olsm.predict(xc_val),0) #adjust round depending on predictions

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred

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

#Let's see what this gives in a regression forest
x = combined_df.drop(columns = ["country", "actor_1_name", "actor_2_name", "actor_3_name", "director_name", "movie_title", "content_rating",
                           "num_critic_for_reviews", "gross", "num_voted_users", "num_user_for_reviews", "imdb_score", "language", "movie_facebook_likes", "duration"]) #remove irrelvant independent variables & target variable
y = combined_df["movie_facebook_likes"] #extract target variable
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = seed)
print(x_train.shape)
print(x.columns.shape)

#run a regression tree
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(max_depth=1, min_samples_leaf =1, random_state=seed).fit(x_train, y_train)
#show feature importance
pd.DataFrame({'category': x_train.columns,'importance':rfreg.feature_importances_}).set_index('category').sort_values(by = 'importance', ascending = False)

#predict regression forest
array_pred = np.round(rfreg.predict(x_val),0)

y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) #index must be same as original database
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred

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

###### END ML CODE ######

### 4. Convert the dataset of results back to a Spark DataFrame ###

# Read the data into a Pandas DataFrame
pandas_df = pd.read_csv('my_file.csv')

# Convert the Pandas DataFrame to a Spark DataFrame
df = spark.createDataFrame(pandas_df)

# Print the schema of the Spark DataFrame
df.printSchema()

### 5. Write this DataFrame to the same S3 bucket dmacademy-course-assets under the prefix vlerick/<your_name>/ as JSON lines. ###
### It is likely Spark will create multiple files there. ###
# ##That is entirely normal and inherent to the distributed processing character of Spark. ###

# Write the DataFrame to the S3 bucket
df.write.json('s3://my_bucket/data/output_file.json')

### 6. Package this set of code in a Docker image that you must push to the AWS elastic container registry ###
### Links to an external site.(ECR) bearing the name 338791806049.dkr.ecr.eu-west-1.amazonaws.com/vlerick_cloud_solutions ###
### and with a tag that starts with your first name. ###

# Install Docker on your local machine

# Write the code that you want to package in a file named main.py

