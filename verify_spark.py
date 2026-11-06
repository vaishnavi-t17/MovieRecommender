
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("MovieRecommenderSetupTest")
    .master("local[2]")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

print("Spark version:", spark.version)
print("Spark is running successfully!")

spark.range(1, 6).show()

spark.stop()
