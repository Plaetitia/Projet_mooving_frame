import streamlit as st
from streamlit_option_menu import option_menu
import random
import pandas as pd
import numpy as np
import requests
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import re
import ast

# === Données TMDB ===
api_cle = "a7f7dc1fbf46cd68589893611fac0e82"
lang = "fr-FR"
tmdb_base_url = "https://api.themoviedb.org/3"

def get_movie_data_by_imdb_id(imdb_id):
    url = f"{tmdb_base_url}/find/{imdb_id}"
    params = {
        "api_key": api_cle,
        "language": lang,
        "external_source": "imdb_id"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("movie_results"):
            return data["movie_results"][0]
    return None

# === Nettoyage des genres avec regex ===
def clean_genres(genres):
    if isinstance(genres, list):
        return ', '.join(genres)
    elif isinstance(genres, str):
        try:
            parsed = ast.literal_eval(genres)
            if isinstance(parsed, list):
                return ', '.join(parsed)
        except:
            pass
        return re.sub(r"[\[\]']", '', genres)
    return "N/A"

# === Chargement des données ===
@st.cache_data
def load_data():
    df = pd.read_csv('df_imdb_tmdb_v2.csv')
    return df

df = load_data()

# === Interface principale ===
image = Image.open("logo.png") 
with st.container():
    col1, col2 = st.columns([1, 2]) 
    with col1:
        st.image(image)
    with col2:
        st.title("Projet 2 Moving Frame") 

# === Sidebar ===
with st.sidebar:
    st.title("La région cible") 
    st.image("creuse.png")
    st.write("Ce système de recommandation de films est conçu pour un cinéma dans la Creuse.") 

    # Création de la colonne 'release_year' à partir de 'release_date'
    if 'release_year' not in df.columns:
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

    # Filtre par période (années)
    st.subheader("Filtrer par période")
    selected_year_range = st.slider(
        "Choisissez une période de sortie",
        min_value=1975,
        max_value=1995,
        value=(1975, 1995),
        step=1
    )

    st.subheader("Contactez-nous")
    with st.form("reply_form"):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        phone = st.text_area("Laissez-nous un commentaire")
        if st.form_submit_button("Envoyer"):
            st.success("Merci pour votre commentaire !")

# Filtrage du DataFrame (en dehors de la sidebar)
df = df[
    (df['release_year'] >= selected_year_range[0]) & 
    (df['release_year'] <= selected_year_range[1])
]

# Résultat du filtre
st.markdown(f"**{len(df)} films** trouvés entre **{selected_year_range[0]}** et **{selected_year_range[1]}**.")
# === Menu principal ===
selection = option_menu(
    menu_title=None,
    options=["Films à la une", "Recommandation", "Recherche par acteur", "Notre équipe"],
    menu_icon="menu-app",
    default_index=0,
    orientation="horizontal"
)

# === Films à la Une ===
if selection == "Films à la une":
    st.write("# Bienvenue sur notre site!")
    st.write(""" 
        **Moving Frame vous présente les films à la Une.**
        Vous ne trouvez pas votre bonheur ? Pas de soucis ! Nous nous adaptons à vos envies.
        Passez à la page de recommandations.
    """)

    df_filtered = df.dropna(subset=['poster_url', 'title'])
    sample_df = df_filtered.sample(n=9, random_state=random.randint(0, 2500))

    st.write("### Films à la Une")
    cols = st.columns(3)
    for i, (_, row) in enumerate(sample_df.iterrows()):
        with cols[i % 3]:
            st.image(row['poster_url'], caption=row['title'], use_container_width=True)

# === Recommandation de films ===
elif selection == "Recommandation":

    text_cols = ['overview', 'title', 'original_title', 'production_companies_name', 'actor', 'director']
    cat_cols = ['genres', 'original_language', 'production_countries', 'spoken_languages']
    num_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

    text_weights = {
        'title': 0.2,
        'overview': 0.15,
        'original_title': 0.4,
        'production_companies_name': 0.1, 
        'actor':0.2,
        'director':0.2
    }

    @st.cache_data
    def compute_similarity(df):
        text_similarity = np.zeros((len(df), len(df)))
        for col, weight in text_weights.items():
            if col in df.columns:
                tfidf_matrix = TfidfVectorizer().fit_transform(df[col].fillna(''))
                sim = cosine_similarity(tfidf_matrix)
                text_similarity += weight * sim

        one_hot_encoded = pd.get_dummies(df[cat_cols], columns=cat_cols).values
        cat_similarity = cosine_similarity(one_hot_encoded)

        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(df[num_cols])
        num_similarity = cosine_similarity(num_scaled)

        weights = {'text': 0.6, 'categorical': 0.3, 'numerical': 0.1}
        final_similarity = (
            weights['text'] * text_similarity +
            weights['categorical'] * cat_similarity +
            weights['numerical'] * num_similarity
        )
        return final_similarity

    final_similarity = compute_similarity(df)

    def recommend_movies_by_title(title, df, similarity_matrix, top_n=4):
        try:
            idx = df[df['title'].str.lower() == title.lower()].index[0]
            sim_scores = list(enumerate(similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in sim_scores[1:top_n+1]]
            return df.iloc[top_indices]
        except IndexError:
            return pd.DataFrame()

    def get_tmdb_data_from_imdb(imdb_id, api_cle):
        try:
            find_url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={api_cle}&external_source=imdb_id"
            find_response = requests.get(find_url)
            find_data = find_response.json()

            if find_data.get("movie_results"):
                movie = find_data["movie_results"][0]
                tmdb_id = movie["id"]

                detail_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_cle}&language=fr-FR"
                detail_response = requests.get(detail_url)
                detail_data = detail_response.json()

                return {
                    "title": detail_data.get("title", "Titre inconnu"),
                    "overview": detail_data.get("overview", "Pas de description disponible."),
                    "poster_path": f"https://image.tmdb.org/t/p/w500{detail_data.get('poster_path')}" if detail_data.get("poster_path") else None
                }
        except Exception as e:
            st.error(f"Erreur TMDb avec IMDb ID {imdb_id} : {e}")
        return None

    def display_movies_api(movie_df):
        for _, row in movie_df.iterrows():
            imdb_id = row.get('imdb_id')
            if imdb_id:
                tmdb_data = get_tmdb_data_from_imdb(imdb_id, api_cle)
                if tmdb_data:
                    title = tmdb_data.get("title", "Titre non disponible")
                    overview = tmdb_data.get("overview", "Pas de description disponible")
                    poster_url = tmdb_data.get("poster_path")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster_url:
                            st.image(poster_url, width=150)
                        else:
                            st.write("Aucune image disponible")
                    with col2:
                        st.markdown(f"### {title}")
                        st.markdown(f"**Description :** {overview}")
                        st.markdown(f"**Genres :** {clean_genres(row.get('genres', []))}")
                        st.markdown(f"**Acteurs :** {row.get('actor', 'N/A')}")
                        st.markdown(f"**Réalisateur :** {row.get('director', 'N/A')}")
                    st.markdown("---")
                else:
                    st.warning(f"Détails indisponibles pour le film avec IMDB ID : {imdb_id}")

    # === Interface utilisateur
    st.title(" Recommandation de Films")
    st.write("Recherchez un film pour obtenir des recommandations personnalisées.")
    st.write("Pour tester : Choisir le titre en anglais d’un film. Le film doit être sorti entre 1975 et 1995, être d’origine française ou américaine, et appartenir à l’un de ces genres : Comédie, Famille ou Drame. Besoin d’exemples : Aladdin, The Gendarme and the Gendarmettes (Le Gendarme et les Gendarmettes), Home Alone (Maman, j’ai raté l’avion), Rocky…")

    movie_title = st.text_input("Entrez un titre de film")

    if movie_title.strip():
        movie_title_lower = movie_title.lower()
        movie_titles_list = df['title'].tolist()
        suggestions = process.extract(movie_title_lower, movie_titles_list, limit=10)
        suggested_titles = [title for title, _ in suggestions]

        if suggested_titles:
            st.write("### Suggestions de titres :")
            selected_title = st.selectbox("Sélectionnez un titre parmi les suggestions :", suggested_titles)

            if selected_title:
                st.write(f"Vous avez sélectionné : **{selected_title}**")
                recommended = recommend_movies_by_title(selected_title, df, final_similarity, top_n=4)

                if not recommended.empty:
                    st.success(f"Recommandation pour le film : **{selected_title}**")
                    display_movies_api(recommended)
                else:
                    st.warning("Aucune recommandation trouvée.")

# === Recherche par acteur ===
if selection == "Recherche par acteur":
    movie_actor = st.text_input("Entrez le nom d’un acteur")
    if st.button("Recherche par acteur"):
        if movie_actor.strip() == "":
            st.warning("Veuillez entrer un nom d’acteur.")
        else:
            matching_movies = df[df['actor'].str.contains(movie_actor, case=False, na=False)]

            if not matching_movies.empty:
                st.success(f"Films avec **{movie_actor}** :")

                for _, row in matching_movies.head(5).iterrows():
                    imdb_id = row.get("imdb_id")
                    
                    if not imdb_id:
                        st.warning("Aucun identifiant IMDB disponible.")
                        continue

                    movie_data = get_movie_data_by_imdb_id(imdb_id)

                    if movie_data:
                        title = movie_data.get("title", "Titre non disponible")
                        overview = movie_data.get("overview", "Pas de description disponible")
                        poster_path = movie_data.get("poster_path")
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if poster_url:
                                st.image(poster_url, width=150)
                            else:
                                st.write("Aucune image disponible")
                        with col2:
                            st.markdown(f"### {title}")
                            st.markdown(f"**Description :** {overview}")
                            st.markdown(f"**Genres :** {clean_genres(row.get('genres', []))}")
                            actors = row.get('actor', "N/A")
                            st.markdown(f"**Acteurs :** {actors}")
                            st.markdown(f"**Réalisateur :** {row.get('director', 'N/A')}")
                        st.markdown("---")
                    else:
                        st.warning(f"Détails indisponibles pour le film avec IMDB ID : {imdb_id}")
            else:
                st.error("Aucun film trouvé avec cet acteur.")

# === Notre équipe ===
elif selection == "Notre équipe":
    st.write("### L’équipe Moving Frame")
    image5 = Image.open("groupe.jpg") 
    st.image(image5)
