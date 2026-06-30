import cv2
import numpy as np
import pandas as pd
import os

# Cherche les pixels de couleur rouille
def calculer_pct_rouille(img_hsv):
    bas_rouille  = np.array([10, 100, 100])  # V minimum = 100
    haut_rouille = np.array([25, 255, 255])  # V maximum = 255

    masque = cv2.inRange(img_hsv, bas_rouille, haut_rouille)
    pct = np.sum(masque > 0) / masque.size
    return pct


# Détecte les variations brusques d'intensité
def calculer_rugosite(img_grise):
    # Filtre Sobel en X et en Y
    sobel_x = cv2.Sobel(img_grise, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_grise, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude du gradient (pythagore)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    # Moyenne de la rugosité sur toute l'image
    return np.mean(gradient)


# "saturation_moyenne" en HSV
# Les pustules (jaune, orange, brun) sont toujours plus saturées que le vert normal d'une feuille saine, Fonctionne pour TOUS les stades de la maladie
def calculer_saturation_moyenne(img_hsv):
    # Canal V = Valeur/Luminosité (indice 2 dans HSV)
    canal_valeur     = img_hsv[:, :, 2]
    canal_saturation = img_hsv[:, :, 1]

    # On garde uniquement les pixels clairs (pas la terre sombre)
    # La terre a V < 80 donc on l'ignore
    masque_sans_terre = canal_valeur > 80

    saturation_filtree = canal_saturation[masque_sans_terre]

    if len(saturation_filtree) == 0:
        return 0.0

    return np.mean(saturation_filtree)


# -----------------------------------------------
# PROGRAMME PRINCIPAL
# On parcourt les deux dossiers et on construit
# le tableau final
# -----------------------------------------------

dossiers = {
    "dataset/saines":  0,   # label 0 = saine
    "dataset/malades": 1    # label 1 = malade
}

resultats = []

for dossier, label in dossiers.items():
    for nom_fichier in os.listdir(dossier):

        # On ne traite que les images
        if not nom_fichier.endswith((".jpg", ".jpeg", ".png")):
            continue

        chemin = os.path.join(dossier, nom_fichier)
        img_bgr = cv2.imread(chemin)

        if img_bgr is None:
            print(f"⚠️ Image non lisible : {nom_fichier}")
            continue

        # Conversions de l'image
        img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_grise = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Calcul des 3 features
        pct_rouille        = calculer_pct_rouille(img_hsv)
        rugosite           = calculer_rugosite(img_grise)
        saturation_moyenne = calculer_saturation_moyenne(img_hsv)

        # Ajout au tableau
        resultats.append({
            "ID_Image":           nom_fichier,
            "pct_rouille":        pct_rouille,
            "rugosite":           rugosite,
            "saturation_moyenne": saturation_moyenne,
            "label_malade":       label
        })

        print(f"✅ Traité : {nom_fichier}")

# Création du DataFrame Pandas
df = pd.DataFrame(resultats)

# Sauvegarde en CSV pour la suite du TP
df.to_csv("features.csv", index=False)

print("\n  Aperçu du tableau :")
print(df.head(10))
print(f"\nTotal images traitées : {len(df)}")
print(f"Images saines  : {len(df[df['label_malade'] == 0])}")
print(f"Images malades : {len(df[df['label_malade'] == 1])}")