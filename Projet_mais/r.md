# Diagnostique des feuilles de Mais

## Technologie:
- python

## But:
- Creer une application web qui va detecter si une image de feuille de mais est malade ou sains (sois 0 ou 1)

## Partie 1: Featuring Engineering 
### A faire
- Utiliser OpenCV ou scikit-image et NumPy pour etudier les images dans le dossier et en extraire des caracteristique

- **extraction de couleur** :
    - RGB->HSV (cv2.cvtColor(cv2.COLOR_BGR2HSV))
    - Definir un masque de couleur pour isoler les teintes "rouilles"
    - calcul de caracteristique X1 : pct_rouille (Nombre de pixels de rouille / Nombre total de pixels de la feuille)

- **extraction de texture et rugosite** :
    - filtre de sobel pour detecter les contours et variations brusques d'intensite
    - calcul de caracteristiques X2 : rugosite (Variance ou moyenne de l’intensité des gradients de Sobel)

- **extraction d'un feature de mon choix** :
    - ajouter une variable de mon choix
    - variable justifie en intuition agronomique / informatique
    - variable personnelle

***Livrable intermédiaire :*** Un tableau Pandas DataFrame contenant la structure suivante :
[ID_Image | pct_rouille | rugosite | votre_variable | label_malade] (où label_malade
vaut 0 ou 1)


