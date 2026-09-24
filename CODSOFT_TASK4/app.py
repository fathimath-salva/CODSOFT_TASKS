import streamlit as st
from movie_recommender import recommend_movies
import base64


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MovieVerse",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{encoded}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    max-width: 900px;
    padding-top: 45px;
    padding-bottom: 50px;
}}

/* Main headings */
h1, h2, h3 {{
    color: #2b211b !important;
}}

/* Select box */
div[data-baseweb="select"] > div {{
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 8px;
}}

/* Find button */
div.stButton > button {{
    width: 100%;
    height: 55px;
    background-color: #34241b;
    color: white;
    border-radius: 8px;
    border: none;
    font-size: 17px;
    font-weight: 800;
}}

div.stButton > button:hover {{
    background-color: #704a32;
    color: white;
}}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: rgba(255, 253, 248, 0.94);
    border-radius: 10px;
    border: 1px solid rgba(100, 70, 50, 0.25);
    box-shadow: 3px 6px 18px rgba(0, 0, 0, 0.15);
}}

</style>
""",
        unsafe_allow_html=True
    )


# Your actual image name
set_background("movie_background.jpeg")


# =========================================================
# MOVIEVERSE HEADER
# =========================================================

st.markdown(
    """
# 🎬 MOVIEVERSE
""",
    unsafe_allow_html=True
)

st.markdown(
    "*Every movie tells a story. Let AI find your next one.*"
)


# =========================================================
# MOVIE SELECTION
# =========================================================

with st.container(border=True):

    st.subheader("🍿 Pick Your Movie")

    st.write(
        "Choose a movie you love and our AI will find movies "
        "with similar genres, keywords and stories."
    )

    movie_list = [
        "Avatar",
        "Avengers",
        "Titanic",
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "Jurassic Park",
        "The Matrix",
        "Iron Man",
        "Guardians of the Galaxy"
    ]

    selected_movie = st.selectbox(
        "🎞️ Choose a movie:",
        movie_list
    )

    st.write("")

    recommend_button = st.button(
        "🎬 FIND MY NEXT MOVIE"
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

if recommend_button:

    recommendations = recommend_movies(selected_movie)

    st.subheader("🎥 Your Movie Night Picks")

    st.info(
        f"🍿 Because you selected **{selected_movie}**, "
        "here are your AI-powered recommendations."
    )

    if recommendations:

        for i, movie in enumerate(recommendations, start=1):

            with st.container(border=True):

                col1, col2 = st.columns([1, 6])

                with col1:
                    st.markdown(
                        f"## {i:02d}"
                    )

                with col2:
                    st.caption("⭐ AI RECOMMENDATION")

                    st.markdown(
                        f"### 🎬 {movie}"
                    )

                    st.write(
                        "Recommended using content-based similarity."
                    )

    else:

        st.warning(
            "Sorry, no recommendations were found for this movie."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    "### 🎞️ MovieVerse"
)

st.write(
    "Content-Based Movie Recommendation System"
)

st.caption(
    "TF-IDF • Cosine Similarity"
)

st.caption(
    "CodSoft Artificial Intelligence Internship — Task 4"
)