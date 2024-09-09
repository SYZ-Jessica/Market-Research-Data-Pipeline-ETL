import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

logging .basicConfig(format='%(levelname)s: %(message)s', level = logging.DEBUG)
# Initialize a spark session.
def initSpark():
    spark = (
        SparkSession.builder
        .appName('ETLPyspark')
        .getOrCreate()
    )
    return spark

def extract(source):
    spark = initSpark()
    data = spark.read.option('header', True).csv(source)

    logging.debug(f'{source} has been extracted')
    logging.info(f'There are {data.count()} rows and {len(data.columns)} columns')
    return data

def transform(apps, reviews, category, min_rating, min_reviews):
    logging.info(f'Tranforming data to create a dataset with all {category} apps and their corresponding reviews with a rating of at least {min_rating} and {min_reviews} reviews')
    # Drop duplicates
    dropped_apps = apps.dropDuplicates(['App'])
    dropped_reviews = reviews.dropDuplicates()
    # Filter 2 dataframes so that only the specific category is left
    dropped_apps = dropped_apps.filter(dropped_apps.Category == category)
    dropped_reviews = dropped_reviews.join(dropped_apps,'App').select(dropped_reviews.App,dropped_reviews.Sentiment_Polarity)
    # Aggregate the reviews dataframe
    # Sentiment_Polarity: on averahe how extreme are the reviews
    agg_reviews = dropped_reviews.groupBy('App').agg(F.mean('Sentiment_Polarity').alias('Sentiment_Polarity'))
    # Join 2 dataframes
    joined_df = dropped_apps.join(agg_reviews, on = 'App', how = 'left')
    # Filter columns
    filtered_df = joined_df.select('App','Rating','Reviews','Installs','Sentiment_Polarity')
    # type conversion
    # dtypes for Reviews is Object(string), need to convert into int
    filtered_df = filtered_df.withColumn('Reviews',F.col('Reviews').cast(IntegerType()))
    # filter according minimum rating and minimum reviews
    result = filtered_df.filter((F.col('Rating') > min_rating) & (F.col('Reviews') > min_reviews))
    result = result.orderBy(F.col('Rating').desc(), F.col('Reviews').desc())

    logging.debug(f'The transformed dataframe has {result.count()} rows and {len(result.columns)} columns.')
    return result

def load(dataframe):
    dataframe.write.csv('resultPyspark.csv', mode = 'overwrite', header = True)
    print('Finished!')


transformed_df = transform(extract('apps_data.csv'),extract('review_data.csv'),'FOOD_AND_DRINK',4.0,1000)
load(transformed_df)
