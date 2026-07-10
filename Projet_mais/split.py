import numpy as np
import pandas as pd

def calculer_purete(y):
    if len(y) == 0:
        return 0.0

    nb_total  = len(y)
    nb_malades = np.sum(y == 1)
    nb_saines  = np.sum(y == 0)

    prop_malades = nb_malades / nb_total
    prop_saines  = nb_saines  / nb_total

    return max(prop_saines, prop_malades)


def trouver_meilleur_split(X_colonne, y):

    N = len(y)  

    meilleur_seuil  = None
    meilleure_purete = 0.0

    valeurs_triees = np.sort(np.unique(X_colonne))

    for i in range(len(valeurs_triees) - 1):

        seuil = (valeurs_triees[i] + valeurs_triees[i+1]) / 2

        groupe_gauche = y[X_colonne <= seuil]   # ≤ seuil
        groupe_droite = y[X_colonne >  seuil]   # > seuil

        taille_gauche = len(groupe_gauche)
        taille_droite = len(groupe_droite)

        purete_gauche = calculer_purete(groupe_gauche)
        purete_droite = calculer_purete(groupe_droite)

        p_split = (taille_gauche / N) * purete_gauche + (taille_droite / N) * purete_droite

        if p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil   = seuil

    return meilleur_seuil, meilleure_purete


df = pd.read_csv("features.csv")

X = df[["pct_rouille", "rugosite", "saturation_moyenne"]]
y = df["label_malade"].values

print("=" * 50)
print("RECHERCHE DU MEILLEUR SPLIT PAR VARIABLE")
print("=" * 50)

for nom_variable in X.columns:
    X_col = X[nom_variable].values

    seuil, purete = trouver_meilleur_split(X_col, y)

    print(f"\n  Variable : {nom_variable}")
    print(f"   Meilleur seuil  : {seuil:.4f}")
    print(f"   Pureté obtenue  : {purete:.4f} ({purete*100:.1f}%)")

print("\n" + "=" * 50)
print("La variable avec la pureté la plus haute")
print("   est la plus discriminante !")
print("=" * 50)