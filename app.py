import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark import StorageLevel
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession, functions as F

# =========================================================
# 1. PATHS AND WINDOWS CONFIGURATION
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

DATASETS = {
    "MovieLens 1M": ("1m", "ml-1m"),
    "MovieLens 10M": ("10m", "ml-10M100K"),
}

if os.name == "nt":
    default_hadoop = Path(r"C:\hadoop")
    if not os.environ.get("HADOOP_HOME") and (default_hadoop / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(default_hadoop)
    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        hadoop_bin = str(Path(hadoop_home) / "bin")
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

# =========================================================
# 2. STREAMLIT PAGE AND STYLE
# =========================================================
st.set_page_config(page_title="CineMatch | Movie Recommendations", page_icon="🎬", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #111111; color: #f5f5f5; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .cinematch-title { color: #e50914; font-size: 2.7rem; font-weight: 900; letter-spacing: 1px; }
    .cinematch-subtitle { color: #bdbdbd; font-size: 1rem; margin-bottom: 1.5rem; }
    div[data-testid="stMetric"] { background: #1d1d1d; border: 1px solid #343434; padding: 15px; border-radius: 10px; }
    .stButton > button { border-radius: 7px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. SPARK AND DATA LOADING
# =========================================================
@st.cache_resource(show_spinner=False)
def get_spark():
    spark = (
        SparkSession.builder
        .appName("CineMatchWebApplication")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource(show_spinner=False)
def load_artifacts(dataset_key: str):
    """Load the trained ALS model and preprocessed Spark datasets once."""
    matching = [folder for _, (key, folder) in DATASETS.items() if key == dataset_key]
    if not matching:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    dataset_name = matching[0]
    base = DATA_DIR / dataset_name
    model_path = MODEL_DIR / f"als_{dataset_name}"
    ratings_path = base / "ratings_clean"
    movies_path = base / "movies_clean"

    required = [model_path / "metadata", ratings_path, movies_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required project output is missing: {path}\n"
                "Run prepare_data.py and train_model.py successfully first."
            )

    spark = get_spark()
    model = ALSModel.load(str(model_path))
    ratings = spark.read.parquet(str(ratings_path)).persist(StorageLevel.MEMORY_AND_DISK)
    movies = spark.read.parquet(str(movies_path)).persist(StorageLevel.MEMORY_AND_DISK)

    # Model factor IDs are the users the ALS model actually learned.
    user_ids = sorted(int(row["id"]) for row in model.userFactors.select("id").collect())

    metrics_path = REPORT_DIR / f"metrics_{dataset_name}.json"
    metrics = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    return {
        "spark": spark,
        "model": model,
        "ratings": ratings,
        "movies": movies,
        "user_ids": user_ids,
        "metrics": metrics,
        "dataset_name": dataset_name,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_catalog(dataset_key: str):
    """Return movie metadata as a small local catalog for quick search/selectors."""
    data = load_artifacts(dataset_key)
    rows = (
        data["movies"]
        .select("movieId", "title", "genres")
        .orderBy("title")
        .collect()
    )
    return [row.asDict() for row in rows]


@st.cache_data(ttl=3600, show_spinner=False)
def search_movies(dataset_key: str, query: str):
    query = query.strip().casefold()
    if not query:
        return []
    catalog = get_movie_catalog(dataset_key)
    results = [
        row for row in catalog
        if query in str(row.get("title", "")).casefold()
    ]
    return results[:50]


@st.cache_data(ttl=3600, show_spinner=False)
def load_popular_movies(dataset_key: str):
    data = load_artifacts(dataset_key)
    rows = (
        data["ratings"].groupBy("movieId")
        .agg(
            F.count("*").alias("rating_count"),
            F.avg("rating").alias("average_rating"),
        )
        .filter(F.col("rating_count") >= 20)
        .join(data["movies"], on="movieId", how="inner")
        .orderBy(F.desc("rating_count"), F.desc("average_rating"))
        .limit(10)
        .collect()
    )
    return [row.asDict() for row in rows]


# =========================================================
# 4. FACTOR VECTORS
# Used for fast personalised ALS recommendations and movie similarity.
# =========================================================
@st.cache_resource(show_spinner=False)
def get_factor_bundle(dataset_key: str):
    """Cache all user/movie latent vectors once; these are small for MovieLens."""
    data = load_artifacts(dataset_key)
    item_rows = data["model"].itemFactors.select("id", "features").collect()
    user_rows = data["model"].userFactors.select("id", "features").collect()

    item_ids = np.asarray([int(row["id"]) for row in item_rows], dtype=np.int64)
    item_matrix = np.asarray(
        [[float(v) for v in row["features"]] for row in item_rows],
        dtype=np.float32,
    )
    if item_matrix.ndim != 2 or item_matrix.shape[0] == 0:
        raise ValueError("The saved ALS model has no movie-factor vectors.")

    norms = np.linalg.norm(item_matrix, axis=1)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized_matrix = item_matrix / safe_norms[:, None]

    user_factors = {
        int(row["id"]): np.asarray([float(v) for v in row["features"]], dtype=np.float32)
        for row in user_rows
    }
    item_index = {int(movie_id): index for index, movie_id in enumerate(item_ids)}

    return {
        "item_ids": item_ids,
        "item_matrix": item_matrix,
        "normalized_matrix": normalized_matrix,
        "user_factors": user_factors,
        "item_index": item_index,
    }


# =========================================================
# 5. MOVIE-SEED + USER-PERSONALIZED RECOMMENDATIONS
# Recommendations change with both the entered title and the user.
# Only movies sharing at least one genre with the entered title are eligible.
# =========================================================
def resolve_movie_title(catalog, query: str):
    """Resolve typed title text without showing a separate movie dropdown."""
    query = query.strip().casefold()
    if not query:
        return None

    def bare_title(title):
        return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip().casefold()

    # Prefer an exact full-title match or an exact match excluding the year.
    exact = [
        row for row in catalog
        if str(row.get("title", "")).strip().casefold() == query
        or bare_title(row.get("title", "")) == query
    ]
    if exact:
        exact.sort(key=lambda row: (len(str(row.get("title", ""))), str(row.get("title", "")).casefold()))
        return exact[0]

    # For partial input, prefer titles starting with the query, then the shortest
    # matching title. Example: "Toy Story" resolves to the original film first.
    matches = [
        row for row in catalog
        if query in str(row.get("title", "")).casefold()
    ]
    if not matches:
        return None
    matches.sort(
        key=lambda row: (
            not str(row.get("title", "")).casefold().startswith(query),
            len(str(row.get("title", ""))),
            str(row.get("title", "")).casefold(),
        )
    )
    return matches[0]


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_user_recommendations(
    dataset_key: str,
    user_id: int,
    seed_movie_id: int,
    top_n: int = 10,
):
    """Genre-first recommendations personalized to a MovieLens user.

    Ranking uses selected-film genre similarity, the user's historical genre
    ratings, ALS movie-to-movie cosine similarity, and ALS user/movie scores.
    Previously rated movies and the selected seed movie are excluded.
    """
    data = load_artifacts(dataset_key)
    factors = get_factor_bundle(dataset_key)
    catalog = get_movie_catalog(dataset_key)
    user_id = int(user_id)
    seed_movie_id = int(seed_movie_id)

    user_vector = factors["user_factors"].get(user_id)
    if user_vector is None:
        return {"seed": None, "preferred_genres": [], "recommendations": []}

    catalog_by_id = {int(row["movieId"]): row for row in catalog}
    seed = catalog_by_id.get(seed_movie_id)
    if seed is None:
        return {"seed": None, "preferred_genres": [], "recommendations": []}

    def genre_set(value):
        return {
            part.strip().casefold()
            for part in str(value or "").split("|")
            if part.strip() and part.strip().casefold() != "(no genres listed)"
        }

    seed_genres = genre_set(seed.get("genres"))
    if not seed_genres:
        return {"seed": seed, "preferred_genres": [], "recommendations": []}

    # Read this user's history once and derive genre preference averages.
    history = (
        data["ratings"]
        .filter(F.col("userId") == user_id)
        .select("movieId", "rating")
        .collect()
    )
    seen = {int(row["movieId"]) for row in history}
    seen.add(seed_movie_id)
    user_mean = float(np.mean([float(row["rating"]) for row in history])) if history else 3.5

    genre_values = {}
    for rating_row in history:
        movie = catalog_by_id.get(int(rating_row["movieId"]))
        if not movie:
            continue
        for genre in genre_set(movie.get("genres")):
            genre_values.setdefault(genre, []).append(float(rating_row["rating"]))

    genre_averages = {
        genre: (sum(values) + 5.0 * user_mean) / (len(values) + 5.0)
        for genre, values in genre_values.items()
    }
    # 0.5 is neutral; genres the user rates above their own average score higher.
    user_genre_affinity = {
        genre: float(np.clip(0.5 + (avg - user_mean) / 2.0, 0.0, 1.0))
        for genre, avg in genre_averages.items()
    }
    preferred_genres = sorted(
        genre_averages,
        key=lambda genre: (genre_averages[genre], len(genre_values[genre]), genre),
        reverse=True,
    )[:3]

    item_ids = factors["item_ids"]
    item_matrix = factors["item_matrix"]
    normalized_matrix = factors["normalized_matrix"]
    item_index = factors["item_index"]

    # Normalize the user's ALS preference to a candidate-ranking percentile.
    user_raw_scores = item_matrix @ user_vector
    user_order = np.argsort(user_raw_scores)
    user_percentiles = np.zeros(len(user_raw_scores), dtype=np.float32)
    if len(user_order) > 1:
        user_percentiles[user_order] = np.linspace(0.0, 1.0, len(user_order), dtype=np.float32)
    elif len(user_order) == 1:
        user_percentiles[user_order] = 1.0
    user_percentile_by_id = {
        int(movie_id): float(user_percentiles[idx])
        for idx, movie_id in enumerate(item_ids)
    }

    # Selected movie's latent similarity differentiates movies with the same tags.
    seed_index = item_index.get(seed_movie_id)
    if seed_index is not None:
        seed_latent_scores = normalized_matrix @ normalized_matrix[seed_index]
        seed_similarity_by_id = {
            int(movie_id): float(np.clip(seed_latent_scores[idx], 0.0, 1.0))
            for idx, movie_id in enumerate(item_ids)
        }
    else:
        seed_similarity_by_id = {}

    scored = []
    for candidate_id, candidate in catalog_by_id.items():
        if candidate_id in seen:
            continue

        candidate_genres = genre_set(candidate.get("genres"))
        if not candidate_genres:
            continue

        shared = seed_genres & candidate_genres
        # Strict content filter: recommendations must share at least one genre
        # with the movie entered by the user.
        if not shared:
            continue

        union = seed_genres | candidate_genres
        jaccard = len(shared) / len(union) if union else 0.0
        seed_coverage = len(shared) / len(seed_genres) if seed_genres else 0.0
        exact_genre_match = candidate_genres == seed_genres

        candidate_genres_with_history = [
            user_genre_affinity[genre]
            for genre in candidate_genres
            if genre in user_genre_affinity
        ]
        personal_genre_score = (
            float(np.mean(candidate_genres_with_history))
            if candidate_genres_with_history else 0.5
        )

        latent_score = seed_similarity_by_id.get(candidate_id, 0.0)
        user_als_score = user_percentile_by_id.get(candidate_id, 0.5)

        # The movie's genre match drives the recommendations, while its ALS
        # similarity and the selected user's history personalize the ordering.
        combined = (
            0.34 * jaccard
            + 0.13 * seed_coverage
            + 0.13 * float(exact_genre_match)
            + 0.22 * latent_score
            + 0.10 * personal_genre_score
            + 0.08 * user_als_score
        )
        scored.append({
            "movieId": candidate_id,
            "title": candidate["title"],
            "genres": candidate.get("genres") or "Unknown",
            "shared_genres": ", ".join(sorted(shared)),
            "genre_match": "Exact genre set" if exact_genre_match else "Shared genres",
            "match_score": round(100.0 * combined, 1),
            "_score": combined,
            "_latent": latent_score,
            "_jaccard": jaccard,
        })

    # Rank primarily by total score, then by selected-movie similarity.
    # Do not sort all exact genre sets ahead of all others: that would make
    # different movies with the same tags produce nearly identical lists.
    scored.sort(
        key=lambda row: (row["_score"], row["_latent"], row["_jaccard"], row["title"].casefold()),
        reverse=True,
    )

    recommendations = [
        {
            "movieId": row["movieId"],
            "title": row["title"],
            "genres": row["genres"],
            "shared_genres": row["shared_genres"],
            "genre_match": row["genre_match"],
            "match_score": row["match_score"],
        }
        for row in scored[:max(0, int(top_n))]
    ]

    return {
        "seed": {"movieId": seed_movie_id, "title": seed["title"], "genres": seed.get("genres") or "Unknown"},
        "preferred_genres": preferred_genres,
        "recommendations": recommendations,
    }


# =========================================================
# 7. ANALYTICS - runs only when the dashboard page is selected
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_analytics(dataset_key: str):
    data = load_artifacts(dataset_key)
    ratings = data["ratings"]
    movies = data["movies"]
    summary = ratings.agg(
        F.count("*").alias("rating_count"),
        F.countDistinct("userId").alias("user_count"),
        F.avg("rating").alias("average_rating"),
    ).first()
    movie_count = movies.select("movieId").distinct().count()
    distribution = [
        row.asDict() for row in ratings.groupBy("rating").count().orderBy("rating").collect()
    ]
    movie_genres = (
        movies.select("movieId", F.explode(F.split(F.col("genres"), r"\|")).alias("genre"))
        .filter(F.col("genre") != "(no genres listed)")
    )
    genre_rows = (
        ratings.join(movie_genres, on="movieId", how="inner")
        .groupBy("genre")
        .agg(F.count("*").alias("rating_count"), F.avg("rating").alias("average_rating"))
        .orderBy(F.desc("rating_count"))
        .limit(15)
        .collect()
    )
    return {
        "rating_count": int(summary["rating_count"]),
        "user_count": int(summary["user_count"]),
        "movie_count": int(movie_count),
        "average_rating": float(summary["average_rating"]),
        "rating_distribution": distribution,
        "genre_stats": [row.asDict() for row in genre_rows],
    }


# =========================================================
# 8. SIDEBAR NAVIGATION AND DATASET CHECKS
# =========================================================
st.sidebar.markdown("## 🎬 CineMatch")
st.sidebar.caption("Big Data Movie Recommendation System")

available_datasets = {}
for label, (key, folder_name) in DATASETS.items():
    model_path = MODEL_DIR / f"als_{folder_name}"
    data_path = DATA_DIR / folder_name
    if (
        (model_path / "metadata").exists()
        and (data_path / "ratings_clean").exists()
        and (data_path / "movies_clean").exists()
    ):
        available_datasets[label] = key

if not available_datasets:
    st.title("🎬 CineMatch")
    st.error("No prepared dataset and trained model were found.")
    st.code(
        "python .\\src\\prepare_data.py --dataset 1m\n"
        "python .\\src\\train_model.py --dataset 1m --user-id 1",
        language="powershell",
    )
    st.stop()

dataset_label = st.sidebar.selectbox("Choose dataset", list(available_datasets.keys()))
dataset_key = available_datasets[dataset_label]
selected_page = st.sidebar.radio(
    "Navigate",
    ["🔎 Discover Movies", "✨ For You", "📊 Analytics Dashboard"],
)

st.markdown('<div class="cinematch-title">CINEMATCH</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cinematch-subtitle">Movie recommendations tailored to your chosen title and user preferences '
    'powered by Apache Spark MLlib ALS.</div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading model and data (first startup only)..."):
        data = load_artifacts(dataset_key)
except Exception as error:
    st.error("Could not load the project data or ALS model.")
    st.exception(error)
    st.stop()

# =========================================================
# 9. DISCOVER MOVIES
# =========================================================
if selected_page == "🔎 Discover Movies":
    st.header("Find your next favorite movie")
    query = st.text_input(
        "Search movie titles",
        placeholder="Try Toy Story, Batman, Star Wars...",
        key="movie_search",
    )
    if query.strip():
        with st.spinner("Searching the local movie catalog..."):
            results = search_movies(dataset_key, query)
        if results:
            st.write(f"Found {len(results)} matching movies (maximum 50).")
            st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
            st.info("To get genre-matched recommendations, open **For You**, enter a movie title and user ID, then submit the form.")
        else:
            st.info("No matching movies were found in this dataset.")

    st.subheader("Popular movies")
    with st.spinner("Loading popular movies (first load may take a little time)..."):
        popular_rows = load_popular_movies(dataset_key)
    popular_df = pd.DataFrame(popular_rows)
    if not popular_df.empty:
        popular_df["average_rating"] = popular_df["average_rating"].round(2)
        st.dataframe(
            popular_df[["title", "genres", "rating_count", "average_rating"]].rename(
                columns={
                    "title": "Movie",
                    "genres": "Genres",
                    "rating_count": "Number of ratings",
                    "average_rating": "Average rating",
                }
            ),
            width="stretch",
            hide_index=True,
        )

# =========================================================
# 10. MOVIE + USER-AWARE RECOMMENDATIONS
# =========================================================
elif selected_page == "✨ For You":
    st.header("Recommendations")
    st.write(
        "Enter a movie title and a MovieLens user ID. CineMatch will recommend "
        "unseen movies that share genres with your chosen title, ranked using "
        "movie similarity and the user's rating history."
    )

    user_ids = data["user_ids"]
    catalog = get_movie_catalog(dataset_key)

    if not catalog:
        st.warning("No movie metadata was found for this dataset.")
    elif not user_ids:
        st.warning("No user factors were found in the trained ALS model.")
    else:
        with st.form("movie_user_recommendation_form", clear_on_submit=False):
            movie_query = st.text_input(
                "Movie title",
                placeholder="Type a title, e.g. Toy Story or The Matrix",
                key="recommendation_movie_title_input",
                help="Type a movie title. You do not need to select from a dropdown. Press Enter or click Generate recommendations.",
            )
            default_user = 1 if 1 in user_ids else user_ids[0]
            user_id = st.number_input(
                "MovieLens user ID",
                min_value=1,
                max_value=max(user_ids),
                value=int(default_user),
                step=1,
                key="recommendation_user_id_input",
                help="For MovieLens 1M, user IDs generally range from 1 to 6040.",
            )
            recommendation_count = st.select_slider(
                "Number of recommendations",
                options=[5, 10, 15, 20],
                value=10,
                key="movie_user_recommendation_count",
            )
            submitted = st.form_submit_button(
                "Generate recommendations",
                type="primary",
                use_container_width=False,
            )

        if submitted:
            resolved_movie = resolve_movie_title(catalog, movie_query)
            if not movie_query.strip():
                st.session_state.pop("last_movie_user_recommendation", None)
                st.warning("Enter a movie title first, such as Toy Story or The Matrix.")
            elif resolved_movie is None:
                st.session_state.pop("last_movie_user_recommendation", None)
                st.error(f'No movie title matching "{movie_query.strip()}" was found. Try a longer or more exact title.')
            elif int(user_id) not in set(user_ids):
                st.session_state.pop("last_movie_user_recommendation", None)
                st.warning("This user ID is not present in the trained model. Try another user ID.")
            else:
                try:
                    with st.spinner("Finding movies with matching genres and user preferences..."):
                        result = get_movie_user_recommendations(
                            dataset_key=dataset_key,
                            user_id=int(user_id),
                            seed_movie_id=int(resolved_movie["movieId"]),
                            top_n=int(recommendation_count),
                        )
                    st.session_state["last_movie_user_recommendation"] = {
                        "dataset_key": dataset_key,
                        "user_id": int(user_id),
                        "query": movie_query.strip().casefold(),
                        "seed_movie_id": int(resolved_movie["movieId"]),
                        "result": result,
                    }
                except Exception as error:
                    st.error("Recommendation generation failed.")
                    st.exception(error)

        # Keep the results visible across harmless Streamlit reruns.
        saved = st.session_state.get("last_movie_user_recommendation")
        current_query = movie_query.strip().casefold() if "movie_query" in locals() else ""
        if (
            saved
            and saved.get("dataset_key") == dataset_key
            and saved.get("user_id") == int(user_id)
            and saved.get("query") == current_query
        ):
            result = saved["result"]
            seed = result.get("seed")
            recs = result.get("recommendations", [])
            if seed:
                st.markdown("---")
                st.success(f'Recommendations based on **{seed["title"]}** for MovieLens user {int(user_id)}')
                info_col1, info_col2 = st.columns([1, 2])
                with info_col1:
                    st.markdown("**Selected movie**")
                    st.write(seed["title"])
                with info_col2:
                    st.markdown("**Required matching genres**")
                    st.write(seed.get("genres") or "Genres unavailable")

                if result.get("preferred_genres"):
                    st.caption(
                        "User's strongest genres from rating history: "
                        + ", ".join(result["preferred_genres"])
                    )

                if recs:
                    recs_df = pd.DataFrame(recs)
                    st.dataframe(
                        recs_df[["title", "genres", "shared_genres", "genre_match", "match_score"]].rename(
                            columns={
                                "title": "Recommended movie",
                                "genres": "Genres",
                                "shared_genres": "Genres matching your movie",
                                "genre_match": "Genre match",
                                "match_score": "Match score (%)",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Every result shares at least one genre with the entered title. "
                        "Ranking combines genre overlap, movie-factor similarity and user preferences. "
                        "Match scores are ranking indicators, not probabilities."
                    )
                else:
                    st.info(
                        "No unseen movies with overlapping genres were found for this user and title. "
                        "Try another title or user ID."
                    )

# =========================================================
# 11. ANALYTICS DASHBOARD
# =========================================================
elif selected_page == "📊 Analytics Dashboard":
    st.header("MovieLens analytics")
    with st.spinner("Preparing analytics (the first load can take longer)..."):
        analytics = load_analytics(dataset_key)
        popular_rows = load_popular_movies(dataset_key)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total ratings", f"{analytics['rating_count']:,}")
    c2.metric("Unique users", f"{analytics['user_count']:,}")
    c3.metric("Movies", f"{analytics['movie_count']:,}")
    c4.metric("Average rating", f"{analytics['average_rating']:.2f} / 5")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Rating distribution")
        distribution_df = pd.DataFrame(analytics["rating_distribution"])
        fig_ratings = px.bar(
            distribution_df,
            x="rating",
            y="count",
            labels={"rating": "Rating (stars)", "count": "Number of ratings"},
            title="How users rate movies",
            template="plotly_dark",
        )
        st.plotly_chart(fig_ratings, width="stretch")

    with right:
        st.subheader("Ratings by genre")
        genre_df = pd.DataFrame(analytics["genre_stats"])
        if not genre_df.empty:
            fig_genres = px.bar(
                genre_df.sort_values("rating_count"),
                x="rating_count",
                y="genre",
                orientation="h",
                color="average_rating",
                labels={
                    "rating_count": "Number of ratings",
                    "genre": "Genre",
                    "average_rating": "Average rating",
                },
                title="Genre popularity and average rating",
                template="plotly_dark",
            )
            st.plotly_chart(fig_genres, width="stretch")

    st.subheader("Most-rated movies")
    popular_df = pd.DataFrame(popular_rows)
    if not popular_df.empty:
        popular_df = popular_df.sort_values("rating_count", ascending=True)
        fig_popular = px.bar(
            popular_df,
            x="rating_count",
            y="title",
            orientation="h",
            hover_data=["average_rating", "genres"],
            labels={"rating_count": "Number of ratings", "title": "Movie"},
            title="Top 10 movies by rating volume",
            template="plotly_dark",
        )
        st.plotly_chart(fig_popular, width="stretch")

    metrics = data["metrics"]
    if metrics:
        st.subheader("ALS model evaluation")
        left_metric, middle_metric, right_metric = st.columns(3)
        if "rmse" in metrics:
            left_metric.metric("RMSE", f"{metrics['rmse']:.4f}")
        if "mae" in metrics:
            middle_metric.metric("MAE", f"{metrics['mae']:.4f}")
        if "training_seconds" in metrics:
            right_metric.metric("Training time", f"{metrics['training_seconds']:.1f} sec")

st.divider()
st.caption(
    "CineMatch | Python + Apache Spark MLlib ALS + Streamlit + Plotly | MovieLens research data"
)
