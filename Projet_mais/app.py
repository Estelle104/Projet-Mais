import streamlit as st
import cv2
import numpy as np
import pickle
import os
import shutil
import pandas as pd
from PIL import Image

from features import (
    calculer_masque_feuille,
    calculer_pct_rouille,
    calculer_rugosite,
    calculer_saturation_moyenne,
    corriger_prediction
)

st.set_page_config(
    page_title="Diagnostic Rouille Mais",
    page_icon=" ",
    layout="wide"
)

@st.cache_resource
def charger_modele():
    with open("modele_foret.pkl", "rb") as f:
        modele = pickle.load(f)
    return modele

modele = charger_modele()

# creer upload
DOSSIER_UPLOADS = "uploads"
os.makedirs(DOSSIER_UPLOADS, exist_ok=True)


def extraire_features(img_bgr):
    img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_grise = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    masque_feuille = calculer_masque_feuille(img_hsv)

    pct_rouille        = calculer_pct_rouille(img_hsv, masque_feuille)
    rugosite           = calculer_rugosite(img_grise, masque_feuille)
    saturation_moyenne = calculer_saturation_moyenne(img_hsv, masque_feuille)

    return np.array([[pct_rouille, rugosite, saturation_moyenne]])


# SIDEBAR — Barre laterale
with st.sidebar:
    st.title("  Diagno-Katsaka")
    st.markdown("""
    **Modele utilise :**
    Random Forest (Scikit-Learn)
    Accuracy : 93.17%
    Rappel   : 89.69%
    """)

# PAGE PRINCIPALE
st.title("  Diagnostic de la Rouille Polysora")

st.divider()

#  UPLOAD ET PReDICTION
st.header(" Analyser une nouvelle feuille")

image_uploadee = st.file_uploader(
    "Choisissez une image de feuille de mais",
    type=["jpg", "jpeg", "png"]
)

if image_uploadee is not None:

    # Affichage de l'image uploadee
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Image uploadee")
        image_pil = Image.open(image_uploadee)
        st.image(image_pil, use_column_width=True)

    # Conversion pour OpenCV
    image_np  = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Extraction des features
    with st.spinner(" Analyse en cours..."):
        features = extraire_features(image_bgr)

    # Prediction
    prediction_modele = int(modele.predict(features)[0])
    prediction, correction_appliquee = corriger_prediction(
        prediction_modele,
        features
    )
    probabilites = None
    if hasattr(modele, "predict_proba"):
        probabilites = modele.predict_proba(features)[0]

    with col2:
        st.subheader("  Resultats de l'analyse")

        # Affichage des features extraites
        st.markdown("**Features extraites :**")
        st.metric(" % Rouille",
                  f"{features[0][0]*100:.2f}%")
        st.metric(" Rugosite",
                  f"{features[0][1]:.2f}")
        st.metric(" Saturation moyenne",
                  f"{features[0][2]:.2f}")

        st.divider()

        # Affichage du diagnostic
        st.subheader("  Diagnostic")
        if probabilites is not None:
            confiance = probabilites[int(prediction_modele)]
            st.metric("Confiance du modele", f"{confiance*100:.1f}%")

        if prediction == 1:
            st.error("  ATTENTION : Feuille Malade\n"
                     "(Rouille Polysora Detectee)")
            st.markdown(" **Action recommandee :** "
                        "Appliquer un traitement fongicide "
                        "et isoler la zone affectee.")
        else:
            st.success(" Feuille Saine\n"
                       "Aucun signe de rouille detecte.")
            st.markdown(" **Action recommandee :** "
                        "Continuer la surveillance reguliere.")
            if correction_appliquee:
                st.info("Correction appliquee : le modele hesitait, mais "
                        "le pourcentage de rouille detecte est trop faible "
                        "pour signaler une rouille.")

    # Sauvegarde dans l'historique
    chemin_sauvegarde = os.path.join(
        DOSSIER_UPLOADS,
        f"{prediction}_{image_uploadee.name}"
    )
    with open(chemin_sauvegarde, "wb") as f:
        f.write(image_uploadee.getbuffer())

    st.success(f" Image sauvegardee dans l'historique !")
    
# TABLEAU COMPARATIF DES MODÈLES
st.divider()
st.header(" Comparaison des modèles")

try:
    df_resultats = pd.read_csv("resultats_comparaison.csv", index_col="Modèle")

    df_resultats["Moyenne"] = df_resultats.mean(axis=1, numeric_only=True)

    st.dataframe(
        df_resultats.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Rappel": "{:.2%}",
            "Moyenne": "{:.2%}"
        }),
        use_container_width=True
    )
except FileNotFoundError:
    st.warning("Le fichier resultats_comparaison.csv est introuvable. "
               "Lancez d'abord le script d'entraînement.")
    
#  GALERIE HISTORIQUE
st.divider()
st.header(" Galerie des analyses precedentes")

images_historique = [
    f for f in os.listdir(DOSSIER_UPLOADS)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(images_historique) == 0:
    st.info("Aucune analyse effectuee pour l'instant. "
            "Uploadez une image ci-dessus !")
else:
    st.markdown(f"**{len(images_historique)} analyse(s) "
                f"effectuee(s)**")

    nb_colonnes = 3
    colonnes = st.columns(nb_colonnes)

    for i, nom_image in enumerate(images_historique):
        chemin = os.path.join(DOSSIER_UPLOADS, nom_image)

        label = int(nom_image.split("_")[0])

        with colonnes[i % nb_colonnes]:
            st.image(chemin, use_column_width=True)
            if label == 1:
                st.error(" Malade")
            else:
                st.success(" Saine")
            st.caption(nom_image[2:]) 

    
    st.divider()
    if st.button(" Vider l'historique"):
        shutil.rmtree(DOSSIER_UPLOADS)
        os.makedirs(DOSSIER_UPLOADS, exist_ok=True)
        st.rerun()
