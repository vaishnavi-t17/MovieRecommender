
from pathlib import Path
import argparse
import json
import time

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

DATASETS = {
    "1m": "ml-1m",
    "10m": "ml-10M100K",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Spark ALS recommender"
    )
    parser.add_argument(
        "--dataset",
        choices=["1m", "10m"],
        default="1m",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="User ID for the example recommendations",
    )
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--reg-param", type=float, default=0.1)
    args = parser.parse_args()

    dataset_name = DATASETS[args.dataset]
    data_dir = PROCESSED_DIR / dataset_name

    train_path = data_dir / "train"
    test_path = data_dir / "test"
    movies_path = data_dir / "movies_clean"
    ratings_path = data_dir / "ratings_clean"

    for path in [
        train_path,
        test_path,
        movies_path,
        ratings_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required processed data not found: {path}\n"
                "Run prepare_data.py successfully first."
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName(f"MovieRecommenderALS_{args.dataset}")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        print(f"\nTraining ALS on MovieLens {args.dataset.upper()}")

        train = spark.read.parquet(str(train_path)).select(
            F.col("userId").cast("int").alias("userId"),
            F.col("movieId").cast("int").alias("movieId"),
            F.col("rating").cast("double").alias("rating"),
        )

        test = spark.read.parquet(str(test_path)).select(
            F.col("userId").cast("int").alias("userId"),
            F.col("movieId").cast("int").alias("movieId"),
            F.col("rating").cast("double").alias("rating"),
        )

        movies = spark.read.parquet(str(movies_path))
        all_ratings = spark.read.parquet(str(ratings_path))

        train_count = train.count()
        test_count = test.count()

        if train_count == 0 or test_count == 0:
            raise ValueError(
                "Training and testing data must both contain rows."
            )

        print(f"Training ratings: {train_count:,}")
        print(f"Testing ratings: {test_count:,}")
        print(f"ALS rank: {args.rank}")
        print(f"Maximum iterations: {args.max_iter}")
        print(f"Regularization parameter: {args.reg_param}")

        # 1. Configure ALS.
        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            rank=args.rank,
            maxIter=args.max_iter,
            regParam=args.reg_param,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42,
        )

        # 2. Train the model.
        start_time = time.perf_counter()
        model = als.fit(train)
        training_seconds = time.perf_counter() - start_time

        print(
            f"\nModel training completed in "
            f"{training_seconds:.2f} seconds."
        )

        # 3. Evaluate on held-out ratings.
        predictions = model.transform(test).dropna(
            subset=["prediction"]
        )

        prediction_count = predictions.count()

        if prediction_count == 0:
            raise ValueError(
                "No test predictions could be evaluated. "
                "Check the training data and user/movie coverage."
            )

        rmse_evaluator = RegressionEvaluator(
            labelCol="rating",
            predictionCol="prediction",
            metricName="rmse",
        )

        mae_evaluator = RegressionEvaluator(
            labelCol="rating",
            predictionCol="prediction",
            metricName="mae",
        )

        rmse = rmse_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)

        print("\nMODEL EVALUATION")
        print("-" * 45)
        print(f"Evaluated predictions: {prediction_count:,}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")

        # Lower RMSE and MAE generally indicate better predictions
        # when comparing models evaluated on the same test set.

        # 4. Save the trained ALS model.
        model_path = MODEL_DIR / f"als_{dataset_name}"

        model.write().overwrite().save(str(model_path))
        print(f"\nModel saved to: {model_path}")

        # 5. Generate personalized recommendations.
        # Only recommend for a user represented in the training set.
        user_df = (
            train.select("userId")
            .distinct()
            .filter(F.col("userId") == args.user_id)
        )

        recommendations_path = (
            REPORT_DIR / f"recommendations_{dataset_name}_user{args.user_id}.csv"
        )

        if user_df.limit(1).count() == 0:
            print(
                f"\nUser {args.user_id} was not found in training data. "
                "Choose another user ID."
            )
            recommendation_count = 0

        else:
            movie_count = movies.select("movieId").distinct().count()
            candidate_count = min(500, movie_count)

            if candidate_count == 0:
                raise ValueError("No movies found in movie metadata.")

            candidate_recommendations = (
                model.recommendForUserSubset(
                    user_df,
                    candidate_count,
                )
                .select(
                    "userId",
                    F.explode("recommendations").alias("rec"),
                )
                .select(
                    "userId",
                    F.col("rec.movieId").alias("movieId"),
                    F.col("rec.rating").alias("predicted_rating"),
                )
            )

            # Remove movies that the user has already rated.
            seen_movies = all_ratings.filter(
                F.col("userId") == args.user_id
            ).select("userId", "movieId").distinct()

            unseen_recommendations = (
                candidate_recommendations
                .join(
                    seen_movies,
                    on=["userId", "movieId"],
                    how="left_anti",
                )
                .join(movies, on="movieId", how="inner")
                .select(
                    "userId",
                    "movieId",
                    "title",
                    "genres",
                    "predicted_rating",
                )
                .orderBy(F.desc("predicted_rating"))
                .limit(10)
            )

            recommendation_count = unseen_recommendations.count()

            print(
                f"\nTOP MOVIE RECOMMENDATIONS FOR USER {args.user_id}"
            )
            print("-" * 65)

            unseen_recommendations.show(
                10,
                truncate=False,
            )

            # Save a portable CSV for inspection.
            # This CSV contains at most 10 rows.
            (
                unseen_recommendations
                .coalesce(1)
                .write
                .mode("overwrite")
                .option("header", "true")
                .csv(str(recommendations_path.with_suffix("")))
            )

            print(
                "Recommendations saved under: "
                f"{recommendations_path.with_suffix('')}"
            )

        # 6. Save evaluation metrics.
        metrics = {
            "dataset": dataset_name,
            "training_rows": train_count,
            "testing_rows": test_count,
            "evaluated_predictions": prediction_count,
            "als_rank": args.rank,
            "max_iterations": args.max_iter,
            "regularization": args.reg_param,
            "rmse": float(rmse),
            "mae": float(mae),
            "training_seconds": training_seconds,
            "recommendation_user_id": args.user_id,
            "recommendation_count": recommendation_count,
        }

        metrics_path = REPORT_DIR / f"metrics_{dataset_name}.json"

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        print(f"\nMetrics saved to: {metrics_path}")
        print("\nALS training and evaluation completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
