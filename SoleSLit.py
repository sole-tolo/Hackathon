import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

Chargement des données

file_path = r"C:\Users\solea\Desktop\Hackathon\okcupid_profiles.csv" df = pd.read_csv(file_path)

file_path2 = r"C:\Users\solea\Desktop\Hackathon\prete.csv" prete = pd.read_csv(file_path2)

🧑🏻‍💻 Code Mise en page

st.title('Bienvenu.es à ROSE BONBON') 
st.write ( " parce qu'en amour, on est tous kitsch...")
video_path = "C:/Users/solea/Desktop/Hackathon/chanson.mp4"
st.video(video_path)
st.title("Tu es seul.e pour cette Saint Valentin? Tu ne sais pas quoi manger, ni quoi écouter.. bref, tu es perdu.e?")
st.header("Ici on t'aide avec quelques tips un peu bidon")
# st.image(lien_image2)
col1, col2,col3 = st.columns([1, 3, 1])
col1.write("")
image_path = "https://www.kideaz.com/wp-content/uploads/elementor/thumbs/moufles-couple-350-450-o8ist9xtyb036k7n6zgp5yv6igi8uvnrzmi24z5c74.jpg"
col2.image(image_path, caption="Your Image", use_column_width=True)
col2.write("")
col3.write("")
}

Lien vers images
lien_image2 = "https://vss.astrocenter.fr/astrocenter/pictures/28716457-oeil.jpg"
st.image(lien_image2))

Fonction de recommandation
🧑🏻‍💻 Usage

# Code de recommandation
def recommandation(orientation_sexuelle,signe_astro, prete):
    astro = True
    # Sélection des colonnes catégoriques
    colonnes_categoriques = ['status', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education',
                             'ethnicity', 'height', 'job', 'last_online', 'location', 'offspring', 'pets', 'religion',
                             'sign', 'smokes', 'speaks']
    st.write("toutes les données sont scientifiques, véridiques et tirées de l'INSEE/CNRS/Matsup. Les numéros de tel ont les donne sur commande (RGPD oblige!)")
    if signe_astro=='aries':
        signe_astro='geminis'
    elif signe_astro=='taurus':
        signe_astro='virgo'
    elif signe_astro=='leo':
        signe_astro='aquarius'
    elif signe_astro=='libra':
        signe_astro='sagitarius'
    elif signe_astro=='cancer':
        signe_astro='scorpio'
    elif signe_astro=='pisces':
        signe_astro='taurus'
    elif signe_astro=='aries':
        signe_astro='capricorn'
    elif signe_astro == 'scorpio':
        signe_astro='leo'
    else:
        astro=False 
        st.image('https://www.napolike.it/wp-content/uploads/2022/12/Madame-1.jpg',width=200)
        st.write("tu n'es pas soumis.e à l'astrologie")
        

    # Filtrage du DataFrame par orientation sexuelle
    df_filtre = prete[prete['orientation'] == orientation_sexuelle]
    if astro==True:
        df_filtre = df_filtre.loc[df_filtre['sign'].str.lower().str.contains(signe_astro)]
    # df_filtre['sign'] = df_filtre['sign'].str.extract(r'(\b\w+\b)')

    # Sélection des colonnes catégoriques dans le DataFrame filtré
    df_cat = df_filtre[colonnes_categoriques]

    # Utilisation de OneHotEncoder pour l'encodage one-hot
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    df_encoded = pd.DataFrame(encoder.fit_transform(df_cat), columns=encoder.get_feature_names_out(colonnes_categoriques))

    # Entraînement du modèle Nearest Neighbors
    k_neighbors = 5  
    nn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean') 
    nn_model.fit(df_encoded)

    # Sélection d'un exemple (premier profil dans le DataFrame filtré)
    exemple = df_encoded.iloc[0].values.reshape(1, -1)

    # Recherche des voisins les plus proches pour l'exemple
    distances, indices = nn_model.kneighbors(exemple)

    # Récupération des voisins dans le DataFrame original
    voisins_originale = df_filtre.iloc[indices[0]]
    # Afficher les informations sur les recommandations dans l'interface utilisateur
    st.dataframe(voisins_originale)
}
🧑🏻‍💻 Interphase utilisateur
st.title("Déjà, trouve ton profil idéal. Pour cela, un peu d'astrologie, un peu d'orientation et BAM!" )
# Vérifier si le formulaire est soumis
with st.form("my_form"):
    # Sélection de l'orientation sexuelle
    choix_orientation = ['straight', 'bisexual', 'gay']
    orientation_sexuelle = st.selectbox('Choisis ton orientation:', choix_orientation)
    choix_signe_astro = ['aries', 'taurus ', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio', 'sagittarius ', 'capricorn', 'aquarius', 'pisces']
    signe_astro = st.selectbox('Choisis ton signe astrologique:', choix_signe_astro)
    # choix_body_type= ['average','fit','athletic','thin','curvy','a little extra','skinny','full figured','overweight','jacked','used up','rather not say']
    # body_type = st.selectbox('Choisis ton corps de reve:',choix_body_type)
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        # Appel à la fonction de recommandation
        recommandation(orientation_sexuelle,signe_astro, prete)
}

🧑🏻‍💻 Choix des cadeaux

file_path3 = r"C:\Users\solea\Desktop\Hackathon\planned_age_2.csv"
planned_age = pd.read_csv(file_path3)

st.title("Maintenant que tu l'as trouvé, que vas-tu lui acheter? Rentre ta tranche d'âge et trouve une idée très originale!")

#  Sélectionner la tranche d'âge
selected_age = st.selectbox("Sélectionner une tranche d'âge", planned_age["age"])

#  Vérifier si la tranche d'âge sélectionnée existe dans le DataFrame
if selected_age in planned_age["age"].values:
    percentage_columns = ['Candy','Flowers','Jewelry','Greeting cards','An evening out','Clothing','Gift cards']
    planned_age[percentage_columns] = planned_age[percentage_columns].apply(lambda x: pd.to_numeric(x.astype(str).str.rstrip('%'), errors='coerce'))
#     # Trouver la catégorie avec le pourcentage le plus élevé pour la tranche d'âge sélectionnée
    selected_row = planned_age[planned_age["age"] == selected_age][percentage_columns].iloc[0]

    min_category = selected_row.idxmin()
    # max_percentage = selected_row[max_category]

    # Afficher les résultats
    st.write(f"Les gens de ton âge achètent rarement des: {min_category}")
    
else:
    st.write(f"La tranche d'âge sélectionnée n'existe pas dans le DataFrame.")

#### et encore TAREK
# Charger les données
lien = r"C:\Users\solea\Desktop\Hackathon\planned_gifts_gender.csv"
planned_gender = pd.read_csv(lien)

# S'assurer que les colonnes d'origine sont de type chaîne de caractères
percentage_columns = ['Candy', 'Flowers', 'Jewelry', 'Greeting cards', 'An evening out', 'Clothing', 'Gift cards']
planned_gender[percentage_columns] = planned_gender[percentage_columns].astype(str)

# Convertir les colonnes de pourcentage en types numériques
planned_gender[percentage_columns] = planned_gender[percentage_columns].apply(lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce'))

# Application Streamlit
st.title("Tu peux aussi choisir en fonction de ton sexe")

# Sélectionner le genre
selected_gender = st.selectbox('Sélectionne ton sexe', planned_gender['Unnamed: 0'])

# Vérifier si le genre sélectionné existe dans le DataFrame
if selected_gender in planned_gender['Unnamed: 0'].values:
    # Sélectionner seulement les colonnes de pourcentage qui sont de type chaîne de caractères
    selected_row = planned_gender[planned_gender['Unnamed: 0'] == selected_gender][percentage_columns].iloc[0]

    # Vérifier si la ligne est vide (aucune valeur numérique)
    if not selected_row.empty: 
        min_category = selected_row.idxmin()
        # max_percentage = selected_row[min_category]

        # Afficher les résultats
        # st.write(f"Pourcentage le plus élevé pour {selected_gender}:")
        st.write(f"Et bien, tu devrais penser à acheter plutôt des: {min_category}")
        # st.write(f"Pourcentage: {max_percentage}")
    else:
        st.write(f"Aucune donnée numérique disponible pour le genre sélectionné.")
else:
    st.write(f"Le genre sélectionné n'existe pas dans le DataFrame.")
}
lien = r"C:\Users\solea\Desktop\Hackathon\planned_gifts_gender.csv"
planned_gender = pd.read_csv(lien)

# S'assurer que les colonnes d'origine sont de type chaîne de caractères
percentage_columns = ['Candy', 'Flowers', 'Jewelry', 'Greeting cards', 'An evening out', 'Clothing', 'Gift cards']
planned_gender[percentage_columns] = planned_gender[percentage_columns].astype(str)

# Convertir les colonnes de pourcentage en types numériques
planned_gender[percentage_columns] = planned_gender[percentage_columns].apply(lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce'))

# Application Streamlit
st.title("Tu peux aussi choisir en fonction de ton sexe")

# Sélectionner le genre
selected_gender = st.selectbox('Sélectionne ton sexe', planned_gender['Unnamed: 0'])

# Vérifier si le genre sélectionné existe dans le DataFrame
if selected_gender in planned_gender['Unnamed: 0'].values:
    # Sélectionner seulement les colonnes de pourcentage qui sont de type chaîne de caractères
    selected_row = planned_gender[planned_gender['Unnamed: 0'] == selected_gender][percentage_columns].iloc[0]

    # Vérifier si la ligne est vide (aucune valeur numérique)
    if not selected_row.empty: 
        min_category = selected_row.idxmin()
        # max_percentage = selected_row[min_category]

        # Afficher les résultats
        # st.write(f"Pourcentage le plus élevé pour {selected_gender}:")
        st.write(f"Et bien, tu devrais penser à acheter plutôt des: {min_category}")
        # st.write(f"Pourcentage: {max_percentage}")
    else:
        st.write(f"Aucune donnée numérique disponible pour le genre sélectionné.")
else:
    st.write(f"Le genre sélectionné n'existe pas dans le DataFrame.")
}

🧑🏻‍💻 Recommandation restaurants par ville et budget

st.title("Et maintenant, pensons à l'estomac. Tu dois l'inviter manger!")

st.write("Questions pour un RESTAU")

link = "Selection_resto_finale.csv"

df_resto = pd.read_csv(link)
df_resto = df_resto.sort_values(by = ['score','ratings'],ascending=False)
noms_nouvelles_colonnes = ['','Id', 'Nom', 'Note', 'Nombre de votes', 'Prix','Adresse','Arabian','African','American','Argentinian','Asian','Australian','BBQ','European','French','Italian','total','ville']
df_resto.columns = noms_nouvelles_colonnes

Choix = st.selectbox(
    'Choisis ton type de cuisine',
    ('Arabian','African','American','Argentinian','Asian','Australian','BBQ','European','French','Italian'))

Ville = st.selectbox(
    'Dans quelle ville veux-tu diner ?',  
('Birmingham',
 'Gardendale',
 'Homewood',
  'Woodlands',
 'Shenandoah',
 'Pinehurst',
 'TOMBALL',
 'Oak Ridge North',
 'Jersey Village')
 )
Budget = st.select_slider(
    'Quel est votre budget ?', 
    options= ['$', '$$', '$$$'])
if Budget == '$':
    st.write('Ok Radin!')
if Budget == '$$':
    st.write('Normal tranquille')
if Budget == '$$$':
    st.write('Ah oui en fait tu veux te mettre à genoux !')

df_resto_selec=df_resto[['Id','Nom','Note','Nombre de votes','Prix','ville', Choix]]
df_resto_selec =df_resto_selec[df_resto_selec[Choix] != 0]
df_resto_selec =df_resto_selec[df_resto_selec['Prix'] <= Budget]
df_resto_selec =df_resto_selec[df_resto_selec['ville']== Ville]
df_resto_selec[['Nom','Note','Nombre de votes','Prix','ville']]

🧑🏻‍💻 Recommandation de musique

st.title('Le plus important maintenant... ton ambiance sonore... pour le ou la faire rêver')
st.write("The perfect sound touch")
link = "tracks_features.csv"
df_musique = pd.read_csv(link)
df_musique =df_musique.dropna()
df_musique = df_musique[['id','name','album','album_id','artists','artist_ids','danceability','energy','loudness','acousticness','liveness','tempo']]

with st.form('formulaire'):
    danse = st.slider('Choisis ton degré de dançabilité?', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    energie = st.slider("Le niveau d'energie?", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    bruit = st.slider("l'intensité sonore?", min_value= -60.0, max_value=8.0, value=-26.0, step=0.01)
    acoustique = st.slider("Tu veux des instrus acoustiques?", min_value= 0.0, max_value=1.0, value=0.50, step=0.001)
    submitted = st.form_submit_button("Submit")

if submitted:

    X = df_musique[['danceability','energy', 'acousticness', 'loudness']]
    y = df_musique['name']

    danceability1 = danse
    energy1 = energie
    loudness1 = bruit
    acousticness1 = acoustique

    mes_datas = np.array([danceability1,energy1,acousticness1,loudness1]).reshape(1,4)
    label_encoder = LabelEncoder()

    label_encoder.fit(y)

    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    knn_model = KNeighborsClassifier()

    knn_model.fit(X_train, y_train)

    predictions = knn_model.predict(mes_datas)
    predicted_song_names = []
    predicted_value = knn_model.kneighbors(mes_datas, n_neighbors=10, return_distance=False)
    st.write(df_musique.iloc[predicted_value[0]][['name','album', 'artists']])

   #message finale
if st.button("Et pour finir, un petit cadeau. Clique ici"):
    # Display a final message and image
    st.header("Toute l'équipe de Rose Bonbon espère t' avoir aidé. Longue vie à la Saint Valentin!")
    path = 'https://42mag.fr/wp-content/uploads/2014/03/07familyawkwarddemo.jpg'
    st.image(path)

### et pour les célib code théo
st.header("Tu es donc seul.e mais tu ne souhaites pas rencontrer quelqu'un. Nous comprenons, tu n'as besoin de personne, tu es parfait.e comme tu es !")

link3 = "dftmdbFrOutSeri.csv"
url = 'https://image.tmdb.org/t/p/original'

# df_musique = pd.read_csv(link)
# df_musique =df_musique.dropna()
id_musique_precis = '7sEYlAeTq6GCSkCV6IbTWi'

df_film = pd.read_csv(link3)
# df_film['poster_path'] = 'https://image.tmdb.org/t/p/original'+ df_film['poster_path']

donnees_musique = df_musique.loc[df_musique['id'] == id_musique_precis, ['name', 'artists']]

st.write('Nous te conseillons une seule musique pendant que tu manges...')
st.write(donnees_musique)

st.write('Et après manger tu peux regarder ces super films !')

tconst_precis = ['tt0243155', 'tt1292566', 'tt1632708','tt6791096','tt1045658']
donnees_films = df_film.loc[df_film['imdb_id'].isin(tconst_precis), ['original_title', 'poster_path']]
# donnees_films

for i in range(len(donnees_films)):
    col1, col2 = st.columns(2)
    col1.image(url + donnees_films.iloc[i]['poster_path'], use_column_width="auto")

    with col2:
        st.header(donnees_films.iloc[i]['original_title'])
}



        
  

    
