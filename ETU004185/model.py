import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def calculer_purete(y):
    if len(y) == 0:
        return 0.0
    
    nb_total   = len(y)
    nb_malades = np.sum(y == 1)
    nb_saines  = np.sum(y == 0)
    
    prop_malades = nb_malades / nb_total
    prop_saines  = nb_saines  / nb_total
    return max(prop_saines, prop_malades)


def trouver_meilleur_split(X, y):
    N = len(y)
    meilleur_seuil   = None
    meilleure_purete = 0.0
    meilleure_variable = None

    for col in range(X.shape[1]):
        X_col = X[:, col]
        valeurs_triees = np.sort(np.unique(X_col))

        for i in range(len(valeurs_triees) - 1):
            seuil = (valeurs_triees[i] + valeurs_triees[i+1]) / 2

            groupe_gauche = y[X_col <= seuil]
            groupe_droite = y[X_col >  seuil]

            taille_gauche = len(groupe_gauche)
            taille_droite = len(groupe_droite)

            purete_gauche = calculer_purete(groupe_gauche)
            purete_droite = calculer_purete(groupe_droite)

            p_split = (taille_gauche / N) * purete_gauche + \
                      (taille_droite / N) * purete_droite

            if p_split > meilleure_purete:
                meilleure_purete   = p_split
                meilleur_seuil     = seuil
                meilleure_variable = col

    return meilleure_variable, meilleur_seuil, meilleure_purete

def construire_arbre(X, y, profondeur=0, max_profondeur=6):

    #  groupe 100% pur
    purete = calculer_purete(y)
    if purete == 1.0:
        prediction = int(y[0])
        return {"prediction": prediction}

    # profondeur maximale atteinte
    if profondeur >= max_profondeur:
        prediction = int(Counter(y).most_common(1)[0][0])
        return {"prediction": prediction}

    # plus assez de donnees
    if len(y) < 2:
        prediction = int(Counter(y).most_common(1)[0][0]) 
        return {"prediction": prediction}

    # meilleur split
    variable, seuil, _ = trouver_meilleur_split(X, y)

    if variable is None:
        prediction = int(Counter(y).most_common(1)[0][0])
        return {"prediction": prediction}

    masque_gauche = X[:, variable] <= seuil
    masque_droite = X[:, variable] >  seuil

    X_gauche, y_gauche = X[masque_gauche], y[masque_gauche]
    X_droite, y_droite = X[masque_droite], y[masque_droite]

    return {
        "variable" : variable,
        "seuil"    : seuil,
        "gauche"   : construire_arbre(X_gauche, y_gauche,
                                      profondeur + 1, max_profondeur),
        "droite"   : construire_arbre(X_droite, y_droite,
                                      profondeur + 1, max_profondeur)
    }


def predire_un(arbre, x):
    if "prediction" in arbre:
        return arbre["prediction"]

    if x[arbre["variable"]] <= arbre["seuil"]:
        return predire_un(arbre["gauche"], x)
    else:
        return predire_un(arbre["droite"], x)


def predire(arbre, X):
    return np.array([predire_un(arbre, x) for x in X])


#  RANDOM FOREST 
def construire_foret(X, y, n_arbres=10, max_profondeur=6):
    foret = []
    N = len(y)

    for i in range(n_arbres):
        indices = np.random.choice(N, size=N, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]

        arbre = construire_arbre(X_sample, y_sample, max_profondeur=max_profondeur)
        
        foret.append(arbre)
        print(f"    Arbre {i+1}/{n_arbres} construit")

    return foret


def predire_foret(foret, X):
    votes = np.array([predire(arbre, X) for arbre in foret])

    predictions = []
    for j in range(X.shape[0]):
        votes_j = votes[:, j]
        prediction = int(Counter(votes_j).most_common(1)[0][0])
        predictions.append(prediction)

    return np.array(predictions)


# CHARGEMENT DES DONNEES
print(" Chargement des donnees")
df = pd.read_csv("features.csv")

X = df[["pct_rouille", "rugosite", "saturation_moyenne"]].values
y = df["label_malade"].values

# 80 train, 20 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f" [] Donnees chargees : {len(X_train)} train, {len(X_test)} test")

# Train
#  Arbre 
print("\n -------- Construction de l'arbre From Scratch...")
arbre_scratch = construire_arbre(X_train, y_train, max_profondeur=3)
pred_arbre_scratch = predire(arbre_scratch, X_test)
print(" [] Arbre From Scratch termine !")

#  Random Forest 
print("\n---------- Construction de la foret From Scratch...")
foret_scratch = construire_foret(X_train, y_train,
                                  n_arbres=10, max_profondeur=3)
pred_foret_scratch = predire_foret(foret_scratch, X_test)
print(" [] Foret From Scratch terminee !")

#  Arbre Scikit-Learn 
print("\n ----------- Entraînement de l'arbre Scikit-Learn...")
arbre_sklearn = DecisionTreeClassifier(criterion="gini", random_state=42)
arbre_sklearn.fit(X_train, y_train)
pred_arbre_sklearn = arbre_sklearn.predict(X_test)
print(" [] Arbre Scikit-Learn termine !")

#  Random Forest Scikit-Learn
print("\n---------Entraînement de la foret Scikit-Learn")
foret_sklearn = RandomForestClassifier(n_estimators=100, random_state=42)
foret_sklearn.fit(X_train, y_train)
pred_foret_sklearn = foret_sklearn.predict(X_test)
print(" [] Foret Scikit-Learn terminee !")

# Comparaisoon
def evaluer(y_test, y_pred, nom):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    return {"Modèle": nom, "Accuracy": acc, "Precision": prec, "Rappel": rec}

resultats = [
    evaluer(y_test, pred_arbre_scratch,  "Arbre From Scratch"),
    evaluer(y_test, pred_foret_scratch,  "Foret From Scratch"),
    evaluer(y_test, pred_arbre_sklearn,  "Arbre Scikit-Learn"),
    evaluer(y_test, pred_foret_sklearn,  "Foret Scikit-Learn"),
]

df_resultats = pd.DataFrame(resultats)
df_resultats = df_resultats.set_index("Modèle")

with open("modele_foret.pkl", "wb") as f:
    pickle.dump(foret_sklearn, f)

print(" [] Modèle sauvegarde dans modele_foret.pkl")

df_resultats.to_csv("resultats_comparaison.csv")
print(" [] Tableau comparatif sauvegarde dans resultats_comparaison.csv")


print("\n")
print("=" * 60)
print("  TABLEAU COMPARATIF DES 4 MODÈLES")
print("=" * 60)
print(df_resultats.to_string())
print("=" * 60)