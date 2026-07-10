import cv2
import numpy as np
import pandas as pd
import os


def corriger_prediction(prediction_modele, features):
    pct_rouille = features[0][0]
    saturation_moyenne = features[0][2]

    feuille_tres_peu_rouillee = pct_rouille < 0.003
    feuille_bien_verte = saturation_moyenne > 130

    if prediction_modele == 1 and feuille_tres_peu_rouillee and feuille_bien_verte:
        return 0, True

    return prediction_modele, False


def calculer_masque_feuille(img_hsv):
    bas_vert_1 = np.array([25, 25, 25])
    haut_vert_1 = np.array([95, 255, 255])
    masque_vert = cv2.inRange(img_hsv, bas_vert_1, haut_vert_1)

    bas_jaune = np.array([15, 35, 35])
    haut_jaune = np.array([35, 255, 255])
    masque_jaune = cv2.inRange(img_hsv, bas_jaune, haut_jaune)

    masque = cv2.bitwise_or(masque_vert, masque_jaune)
    noyau = np.ones((5, 5), np.uint8)
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)
    return masque > 0


def calculer_pct_rouille(img_hsv, masque_feuille=None):
    bas_rouille  = np.array([10, 100, 100])
    haut_rouille = np.array([25, 255, 255])
    masque = cv2.inRange(img_hsv, bas_rouille, haut_rouille)
    if masque_feuille is None:
        pct = np.sum(masque > 0) / masque.size
    else:
        nb_pixels_feuille = np.sum(masque_feuille)
        if nb_pixels_feuille == 0:
            return 0.0
        pct = np.sum((masque > 0) & masque_feuille) / nb_pixels_feuille
    return pct


def calculer_rugosite(img_grise, masque_feuille=None):
    sobel_x = cv2.Sobel(img_grise, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_grise, cv2.CV_64F, 0, 1, ksize=3)
    # Magnitude du gradient (pythagore)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    if masque_feuille is not None and np.any(masque_feuille):
        return np.mean(gradient[masque_feuille])
    return np.mean(gradient)


def calculer_saturation_moyenne(img_hsv, masque_feuille=None):
    canal_valeur      = img_hsv[:, :, 2]
    canal_saturation  = img_hsv[:, :, 1]
    masque_sans_terre = canal_valeur > 80
    if masque_feuille is not None:
        masque_sans_terre = masque_sans_terre & masque_feuille
    saturation_filtree = canal_saturation[masque_sans_terre]
    if len(saturation_filtree) == 0:
        return 0.0
    return np.mean(saturation_filtree)


if __name__ == "__main__":

    dossiers = {
        "dataset/saines":  0,
        "dataset/malades": 1
    }

    resultats = []

    for dossier, label in dossiers.items():
        for nom_fichier in os.listdir(dossier):

            if not nom_fichier.endswith((".jpg", ".jpeg", ".png")):
                continue

            chemin  = os.path.join(dossier, nom_fichier)
            img_bgr = cv2.imread(chemin)

            if img_bgr is None:
                print(f"Image non lisible : {nom_fichier}")
                continue

            # Conversions de l'image
            img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_grise = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            masque_feuille = calculer_masque_feuille(img_hsv)

            # Calcul des 3 features
            pct_rouille        = calculer_pct_rouille(img_hsv, masque_feuille)
            rugosite           = calculer_rugosite(img_grise, masque_feuille)
            saturation_moyenne = calculer_saturation_moyenne(img_hsv, masque_feuille)

            resultats.append({
                "ID_Image":           nom_fichier,
                "pct_rouille":        pct_rouille,
                "rugosite":           rugosite,
                "saturation_moyenne": saturation_moyenne,
                "label_malade":       label
            })

            print(f"[ok] Traite : {nom_fichier}")

    df = pd.DataFrame(resultats)

    df.to_csv("features.csv", index=False)

    print("\nApercu du tableau :")
    print(df.head(10))
    print(f"\nTotal images traitees : {len(df)}")
    print(f"Images saines  : {len(df[df['label_malade'] == 0])}")
    print(f"Images malades : {len(df[df['label_malade'] == 1])}")
