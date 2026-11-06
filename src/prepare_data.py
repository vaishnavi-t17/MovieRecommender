
from pathlib import Path
import argparse

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    LongType,
    StringType,
)

# Automatically locate the project directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

DATASETS = {
    "1m": "ml-1m",
    "10m": "ml-10M100K",
}

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


def read_file(spark, path, schema):
    """Read MovieLens's double-colon-separated files."""
    return (
        spark.read
        .option("sep", "::")
        .option("encoding", "ISO-8859-1")
        .schema(schema)
        .csv(str(path))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean and split MovieLens data with PySpark"
    )
    parser.add_argument(
        "--dataset",
        choices=["1m", "10m"],
        default="1m",
        help="MovieLens dataset to prepare",
    )
    args = parser.parse_args()

    dataset_name = DATASETS[args.dataset]
    input_dir = RAW_DIR / dataset_name
    output_dir = PROCESSED_DIR / dataset_name

    ratings_path = input_dir / "ratings.dat"
    movies_path = input_dir / "movies.dat"

    for path in [ratings_path, movies_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Check your data/raw folder."
            )

    spark = (
        SparkSession.builder
        .appName(f"MovieRecommenderPreparation_{args.dataset}")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        print(f"\nPreparing MovieLens {args.dataset.upper()}...")
        print("Reading ratings and movie metadata...")

        ratings_raw = read_file(
            spark, ratings_path, RATINGS_SCHEMA
        )
        movies_raw = read_file(
            spark, movies_path, MOVIES_SCHEMA
        )

        # 1. Identify invalid ratings.
        invalid_condition = (
            F.col("userId").isNull()
            | F.col("movieId").isNull()
            | F.col("rating").isNull()
            | (F.col("userId") <= 0)
            | (F.col("movieId") <= 0)
            | (~F.col("rating").between(0.5, 5.0))
        )

        raw_rating_count = ratings_raw.count()
        invalid_rating_count = ratings_raw.filter(
            invalid_condition
        ).count()

        # Keep rows that have valid user IDs, movie IDs and ratings.
        valid_ratings = ratings_raw.filter(
            ~invalid_condition
        )

        # 2. Remove repeated user/movie pairs.
        # If a pair occurs more than once, keep its latest timestamp.
        # Rating descending provides a deterministic tie-breaker.
        rating_window = (
            Window
            .partitionBy("userId", "movieId")
            .orderBy(
                F.col("timestamp").desc_nulls_last(),
                F.col("rating").desc(),
            )
        )

        ratings_clean = (
            valid_ratings
            .withColumn(
                "_row_number",
                F.row_number().over(rating_window),
            )
            .filter(F.col("_row_number") == 1)
            .drop("_row_number")
            .persist(StorageLevel.DISK_ONLY)
        )

        clean_rating_count = ratings_clean.count()
        duplicates_removed = (
            raw_rating_count
            - invalid_rating_count
            - clean_rating_count
        )

        # 3. Clean movie metadata.
        valid_movies = (
            movies_raw
            .filter(
                F.col("movieId").isNotNull()
                & (F.col("movieId") > 0)
                & F.col("title").isNotNull()
                & (F.length(F.trim(F.col("title"))) > 0)
            )
            .withColumn("title", F.trim(F.col("title")))
            .withColumn(
                "genres",
                F.when(
                    F.col("genres").isNull()
                    | (F.length(F.trim(F.col("genres"))) == 0),
                    F.lit("(no genres listed)"),
                ).otherwise(F.trim(F.col("genres"))),
            )
        )

        # Keep one metadata record per movie ID.
        movie_window = (
            Window
            .partitionBy("movieId")
            .orderBy(
                F.col("title").asc(),
                F.col("genres").asc(),
            )
        )

        movies_clean = (
            valid_movies
            .withColumn(
                "_row_number",
                F.row_number().over(movie_window),
            )
            .filter(F.col("_row_number") == 1)
            .drop("_row_number")
        )

        # 4. Split ratings into training and testing datasets.
        # A fixed seed makes the split reproducible.
        train_df, test_df = ratings_clean.randomSplit(
            [0.8, 0.2],
            seed=42,
        )

        train_count = train_df.count()
        test_count = test_df.count()
        movie_count = movies_clean.count()

        if train_count == 0 or test_count == 0:
            raise ValueError(
                "Training or testing data is empty."
            )

        # 5. Save Spark Parquet datasets.
        # Each output path is a directory containing Parquet part files.
        output_dir.mkdir(parents=True, exist_ok=True)

        ratings_clean.write.mode("overwrite").parquet(
            str(output_dir / "ratings_clean")
        )

        movies_clean.write.mode("overwrite").parquet(
            str(output_dir / "movies_clean")
        )

        train_df.write.mode("overwrite").parquet(
            str(output_dir / "train")
        )

        test_df.write.mode("overwrite").parquet(
            str(output_dir / "test")
        )

        print("\n" + "=" * 55)
        print("DATA PREPARATION SUMMARY")
        print("=" * 55)
        print(f"Dataset: MovieLens {args.dataset.upper()}")
        print(f"Original rating rows: {raw_rating_count:,}")
        print(f"Invalid rating rows removed: {invalid_rating_count:,}")
        print(f"Duplicate user/movie pairs removed: {duplicates_removed:,}")
        print(f"Clean rating rows: {clean_rating_count:,}")
        print(f"Clean movie records: {movie_count:,}")
        print(f"Training rows: {train_count:,}")
        print(f"Testing rows: {test_count:,}")

        print("\nAverage rating:")
        ratings_clean.select(
            F.avg("rating").alias("average_rating")
        ).show()

        print("Output directory:")
        print(output_dir)

        print("\nSaved:")
        print("  ratings_clean/")
        print("  movies_clean/")
        print("  train/")
        print("  test/")
        print("\nData preparation completed successfully.")

        ratings_clean.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
