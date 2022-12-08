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
df.write.csv('s3://my_bucket/data/output_file.csv')
df.write.json('s3://my_bucket/data/output_file.json')
df.write.parquet('s3://my_bucket/data/output_file.parquet')

### 6. Package this set of code in a Docker image that you must push to the AWS elastic container registry ###
### Links to an external site.(ECR) bearing the name 338791806049.dkr.ecr.eu-west-1.amazonaws.com/vlerick_cloud_solutions ###
### and with a tag that starts with your first name. ###

# Install Docker on your local machine

# Write the code that you want to package in a file named main.py

# Create a Dockerfile with the following contents:
#
FROM python:3.7
ADD main.py /
RUN pip install pandas
CMD ["python", "./main.py"]

# Build the Docker image using the Dockerfile
$ docker build -t my_image .

# Tag the Docker image with a unique name
$ docker tag my_image <my_aws_account_id>.dkr.ecr.<my_region>.amazonaws.com/my_repository:latest

# Push the Docker image to ECR
$ docker push <my_aws_account_id>.dkr.ecr.<my_region>.amazonaws.com/my_repository:latest




