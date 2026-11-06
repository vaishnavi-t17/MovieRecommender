from pathlib import Path
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    LongType,
    StringType,
)

# Find the root project folder automatically.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

# MovieLens files do not have column headers.
RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), True),
    StructField("movieId", IntegerType(), True),
    StructField("rating", DoubleType(), True),
    StructField("timestamp", LongType(), True),
])

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), True),
    StructField("title", StringType(), True),
    StructField("genres", StringType(), True),
])

USERS_SCHEMA = StructType([
    StructField("userId", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("occupation", IntegerType(), True),
    StructField("zipCode", StringType(), True),
])

DATASET_PATHS = {
    "1m": RAW_DATA_DIR / "ml-1m",
    "10m": RAW_DATA_DIR / "ml-10M100K",
}


def read_movielens_file(spark, path, schema):
    """Read a MovieLens file using its :: delimiter."""
    return (
        spark.read
        .option("sep", "::")
        .option("encoding", "ISO-8859-1")
        .schema(schema)
        .csv(str(path))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect MovieLens data using Apache Spark"
    )
    parser.add_argument(
        "--dataset",
        choices=["1m", "10m"],
        default="1m",
        help="Choose MovieLens 1M or MovieLens 10M",
    )
    args = parser.parse_args()

    dataset_dir = DATASET_PATHS[args.dataset]
    ratings_path = dataset_dir / "ratings.dat"
    movies_path = dataset_dir / "movies.dat"
    users_path = dataset_dir / "users.dat"

    # Check paths before starting Spark.
    for path in [ratings_path, movies_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Check that the dataset was extracted correctly."
            )

    spark = (
        SparkSession.builder
        .appName("MovieRecommenderDataInspection")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print("\n" + "=" * 55)
        print(f"MOVIELENS {args.dataset.upper()} DATA INSPECTION")
        print("=" * 55)

        # Load ratings and movies into Spark DataFrames.
        ratings = read_movielens_file(
            spark, ratings_path, RATINGS_SCHEMA
        )
        movies = read_movielens_file(
            spark, movies_path, MOVIES_SCHEMA
        )

        # Inspect rating counts and data quality.
        summary = ratings.agg(
            F.count("*").alias("rating_rows"),
            F.countDistinct("userId").alias("unique_users"),
            F.min("rating").alias("minimum_rating"),
            F.max("rating").alias("maximum_rating"),
            F.avg("rating").alias("average_rating"),
            F.sum(
                F.when(
                    F.col("userId").isNull()
                    | F.col("movieId").isNull()
                    | F.col("rating").isNull(),
                    1,
                ).otherwise(0)
            ).alias("rows_with_null_required_fields"),
        ).first()

        movie_count = movies.count()

        print(f"\nRatings rows: {summary['rating_rows']:,}")
        print(f"Unique users in ratings: {summary['unique_users']:,}")
        print(f"Movies: {movie_count:,}")
        print(f"Minimum rating: {summary['minimum_rating']}")
        print(f"Maximum rating: {summary['maximum_rating']}")
        print(f"Average rating: {summary['average_rating']:.3f}")
        print(
            "Rows with missing required fields: "
            f"{summary['rows_with_null_required_fields']:,}"
        )

        # Show actual records from the dataset.
        print("\nSAMPLE RATINGS")
        ratings.show(5, truncate=False)

        print("\nSAMPLE MOVIES")
        movies.show(5, truncate=False)

        print("\nRATING DISTRIBUTION")
        ratings.groupBy("rating").count().orderBy("rating").show()

        # Demographic data is available in MovieLens 1M.
        # It is not included in the MovieLens 10M dataset.
        if users_path.exists():
            users = read_movielens_file(
                spark, users_path, USERS_SCHEMA
            )
            print(f"\nUser demographic records: {users.count():,}")
            print("\nSAMPLE USERS")
            users.show(5, truncate=False)
        else:
            print(
                "\nNo users.dat file in this dataset. "
                "User counts are derived from ratings."
            )

        print("\nData inspection completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()