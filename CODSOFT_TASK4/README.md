# 🎬 Movie Recommendation System

## 📌 Project Overview

This project is a simple Movie Recommendation System developed as part of the CodSoft Artificial Intelligence Internship.

The system recommends movies based on the similarity of their genres, keywords, and descriptions.

## 🧠 Technique Used

This project uses **Content-Based Filtering**.

The system uses:

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit

## ⚙️ How It Works

1. The movie dataset is loaded using Pandas.
2. Movie features such as genre, keywords, and overview are combined.
3. TF-IDF converts the text information into numerical vectors.
4. Cosine Similarity calculates how similar the movies are.
5. The system displays the top 5 similar movies.

## 🚀 How to Run

Install the required libraries:

```bash
pip install -r requirements.txt