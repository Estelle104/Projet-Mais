import numpy as np
import pandas as pd

# Calculer la pureté d'un groupe
# P(t) = proportion de la classe majoritaire
def calculer_purete(y):
    # Si le groupe est vide, pureté = 0
    if len(y) == 0:
        return 0.0

    # Nombre de chaque classe
    nb_total  = len(y)
    nb_malades = np.sum(y == 1)
    nb_saines  = np.sum(y == 0)

    # Proportion de chaque classe
    prop_malades = nb_malades / nb_total
    prop_saines  = nb_saines  / nb_total

    # On retourne la proportion de la classe majoritaire
    return max(prop_saines, prop_malades)


def trouver_meilleur_split(X_colonne, y):
    # X_colonne : les valeurs d'une feature (ex: pct_rouille)
    # y         : les labels (0 = saine, 1 = malade)

    N = len(y)  

    meilleur_seuil  = None
    meilleure_purete = 0.0

    # Étape 1 : on trie les valeurs de X
    valeurs_triees = np.sort(np.unique(X_colonne))

    # Étape 2 : on teste chaque seuil possible
    for i in range(len(valeurs_triees) - 1):

        #milieu entre deux valeurs consécutives
        seuil = (valeurs_triees[i] + valeurs_triees[i+1]) / 2

        # Étape 3 : on sépare en deux groupes
        groupe_gauche = y[X_colonne <= seuil]   # ≤ seuil
        groupe_droite = y[X_colonne >  seuil]   # > seuil

        # Taille des deux groupes
        taille_gauche = len(groupe_gauche)
        taille_droite = len(groupe_droite)

        # Pureté de chaque groupe
        purete_gauche = calculer_purete(groupe_gauche)
        purete_droite = calculer_purete(groupe_droite)

        # Pureté pondérée du split
        # P_split = (|G|/N) × P(G) + (|D|/N) × P(D)
        p_split = (taille_gauche / N) * purete_gauche + (taille_droite / N) * purete_droite

        # Étape 4 : on garde le meilleur seuil
        if p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil   = seuil

    return meilleur_seuil, meilleure_purete


# TEST : On charge notre fichier features.csv
# et on teste la fonction sur chaque variable
df = pd.read_csv("features.csv")

# On sépare les features et les labels
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
print("✅ La variable avec la pureté la plus haute")
print("   est la plus discriminante !")
print("=" * 50)