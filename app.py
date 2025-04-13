import streamlit as st
from streamlit_option_menu import option_menu
import random
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import re
import ast

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

    st.subheader("Contactez-nous")
    with st.form("reply_form"):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        phone = st.text_input("Téléphone")
        message = st.text_area("Ecrivez votre message")
        if st.form_submit_button("Envoyer"):
            st.success("Merci pour votre réponse!")

    st.subheader("Laissez-nous un commentaire")
    with st.form("comment_form"):
        comment = st.text_area("Votre commentaire")
        if st.form_submit_button("Poster"):
            st.success("Commentaire posté avec succès!") 

# === Menu principal ===
selection = option_menu(
    menu_title=None,
    options=["Les films à la une", "Recommandation", "Recherche par acteurs", "Notre équipe"],
    menu_icon="menu-app",
    default_index=0,
    orientation="horizontal"
)

# === Films à la Une ===
if selection == "Les films à la une":
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
    text_cols = ['overview', 'title', 'original_title', 'production_companies_name']
    cat_cols = ['genres', 'original_language', 'production_countries', 'spoken_languages']
    num_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']

    text_weights = {
        'title': 0.2,
        'overview': 0.15,
        'original_title': 0.4,
        'production_companies_name': 0.1
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

    def display_movies(movie_df):
        for _, row in movie_df.iterrows():
            st.markdown(f"### {row['original_title']}")
            if pd.notna(row.get('poster_url', None)):
                st.image(row['poster_url'], width=200)
            st.write(f"**Genres :** {clean_genres(row.get('genres', []))}")
            st.write(f"**Acteurs :** {row.get('actor', 'N/A')}")
            st.write(f"**Réalisateur :** {row.get('director', 'N/A')}")
            st.markdown("---")

    st.title("Recommandation de Films")
    st.write("Recherchez un film par titre ou acteur pour obtenir des recommandations personnalisées.")

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
                    for _, row in recommended.iterrows():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if pd.notna(row.get('poster_url', None)):
                                st.image(row['poster_url'], width=150)
                            else:
                                st.write("Aucune image disponible")
                        with col2:
                            st.markdown(f"### {row.get('original_title', 'Titre inconnu')}")
                            st.markdown(f"**Description :** {row.get('overview', 'Aucune description disponible')}")
                            st.markdown(f"**Genres :** {clean_genres(row.get('genres', []))}")
                            actors = row.get('actor', [])
                            if isinstance(actors, str):
                                actors = actors
                            elif isinstance(actors, list):
                                actors = ', '.join(actors)
                            st.markdown(f"**Acteurs :** {actors}")
                            st.markdown(f"**Réalisateur :** {row.get('director', 'N/A')}")
                        st.markdown("---")
                else:
                    st.error("Aucune recommandation trouvée.")
        else:
            st.write("Aucune suggestion trouvée.")
    else:
        st.warning("Veuillez entrer un titre.")

# === Recherche par acteur ===
elif selection == "Recherche par acteurs":
    movie_actor = st.text_input("Entrez le nom d’un acteur")
    if st.button("Recommander par acteur"):
        if movie_actor.strip() == "":
            st.warning("Veuillez entrer un nom d’acteur.")
        else:
            matching_movies = df[df['actor'].str.contains(movie_actor, case=False, na=False)]
            if not matching_movies.empty:
                st.success(f"Films avec **{movie_actor}** :")
                for _, row in matching_movies.head(5).iterrows():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if pd.notna(row.get('poster_url', None)):
                            st.image(row['poster_url'], width=150)
                        else:
                            st.write("Aucune image disponible")
                    with col2:
                        st.markdown(f"### {row.get('original_title', 'Titre inconnu')}")
                        st.markdown(f"**Description :** {row.get('overview', 'Aucune description disponible')}")
                        st.markdown(f"**Genres :** {clean_genres(row.get('genres', []))}")
                        
                        actors = row.get('actor', [])
                        if isinstance(actors, str):
                            pass
                        elif isinstance(actors, list):
                            actors = ', '.join(actors)
                        else:
                            actors = "N/A"
                        st.markdown(f"**Acteurs :** {actors}")
                        st.markdown(f"**Réalisateur :** {row.get('director', 'N/A')}")
                    st.markdown("---")
            else:
                st.error("Aucun film trouvé avec cet acteur.")

# === Notre équipe ===
elif selection == "Notre équipe":
    st.write("### L’équipe Moving Frame")
    image5 = Image.open("groupe.jpg") 
    st.image(image5)
