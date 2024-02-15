import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

# Chargement des données

file_path = r"C:\Users\solea\Desktop\Hackathon\okcupid_profiles.csv"
df = pd.read_csv(file_path)

file_path2 = r"C:\Users\solea\Desktop\Hackathon\prete.csv"
prete = pd.read_csv(file_path2)

## 🧑🏻‍💻 Code Mise en page 

```js
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
```
        
  

    
