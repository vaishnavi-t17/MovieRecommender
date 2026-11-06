# 🎬 Movie Recommender System

A movie recommendation system built using **Python, Apache Spark (PySpark), Collaborative Filtering, and Streamlit**.

This project explores recommendation generation using the **Alternating Least Squares (ALS)** algorithm to recommend movies based on user preferences and interaction patterns.

## 🚀 Technologies Used

- **Python** – Core programming language
- **Apache Spark (PySpark)** – Distributed data processing
- **ALS (Alternating Least Squares)** – Collaborative filtering recommendation algorithm
- **Streamlit** – Interactive web application
- **Pandas & NumPy** – Data processing and numerical operations

## 📂 Project Structure

```text
MovieRecommender/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── verify_spark.py         # Spark environment verification
│
├── src/
│   ├── inspect_data.py     # Dataset inspection
│   ├── prepare_data.py     # Data preparation
│   └── train_model.py      # Model training
│
├── models/
│   ├── als_ml-1m/          # ALS model output
│   └── als_ml-10M100K/     # ALS model output
│
├── reports/
│   ├── metrics_ml-1m.json
│   ├── metrics_ml-10M100K.json
│   └── recommendations/
│
└── .gitignore
```

## ⚙️ Prerequisites

- Python 3.11
- Java 17
- Git

## 💻 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/vaishnavi-t17/MovieRecommender.git
cd MovieRecommender
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the application**

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 🧠 Recommendation Algorithm

The project uses **Alternating Least Squares (ALS)**, a collaborative filtering algorithm.

ALS learns latent representations of users and movies from rating interactions and uses them to generate personalized recommendations.

## 📊 Model Outputs

The repository includes generated ALS model outputs and recommendation reports for two dataset configurations:

- MovieLens 1M
- MovieLens 10M100K

## 👩‍💻 Author

**Vaishnavi T**

GitHub: [@vaishnavi-t17](https://github.com/vaishnavi-t17)

---

⭐ If you find this project interesting, feel free to explore the repository!
