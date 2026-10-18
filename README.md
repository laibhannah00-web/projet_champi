# projet_champi

## Objectif

Prédire la **toxicité d'un champignon** (vénéneux ou comestible) à partir de ses caractéristiques, dans le cadre de la compétition Kaggle [UWI Poisonous Mushrooms](https://www.kaggle.com/competitions/uwi-poisonous-mushorooms).

Il s'agit d'un problème de **classification binaire** : pour chaque champignon, le modèle prédit 0 ou 1.

## Les fichiers de la compétition

| Fichier | Rôle |
|---|---|
| `train.csv` | Données d'apprentissage : caractéristiques **et** réponse (vénéneux / comestible) |
| `test.csv` | Données à prédire : caractéristiques **sans** la réponse |
| `sample_submission.csv` | Modèle du format du fichier à envoyer à Kaggle |

## Organisation du projet

Le projet se déroule en 3 phases :

1. **Phase I** : préparer la table `train` et entraîner le modèle
2. **Phase II** : préparer la table `test` et prédire
3. **Phase III** : mettre en forme et envoyer la soumission

---

## Phase I : table `train`

**But** : obtenir un modèle entraîné et vérifié.

### 1. Nettoyage des données (Data Cleaning)

- **Valeurs manquantes** : repérer les cellules vides (`isnull().sum()`), puis les supprimer (`dropna`) ou les remplacer (`fillna`) par la moyenne, la médiane, la valeur la plus fréquente ou via un algorithme.
- **Doublons** : supprimer les lignes redondantes qui pourraient biaiser l'apprentissage (`drop_duplicates`).
- **Valeurs aberrantes (outliers)** : repérer (`describe`, boxplot, méthode IQR) et analyser les valeurs extrêmes (erreurs de mesure ou cas exceptionnels), puis décider de les corriger, les plafonner (`clip`) ou les supprimer.

> Sur `train`, on peut supprimer des lignes. Les valeurs de remplacement et les bornes d'outliers sont **calculées sur `train`** pour être réutilisées sur `test`.

### 2. Traitement des variables catégorielles

Les algorithmes de Machine Learning ne comprennent que les nombres : il faut convertir les textes en valeurs numériques.

- **Encodage ordinal** : pour des catégories avec un ordre logique (ex. faible = 1, moyen = 2, élevé = 3).
- **Encodage One-Hot** : pour des catégories sans ordre (ex. une colonne « couleur » avec rouge, vert, bleu devient plusieurs colonnes 0/1).

### 3. Ingénierie des caractéristiques (Feature Engineering)

- **Création de variables** : combiner des données existantes pour faire ressortir une information plus pertinente (ratio, extraction d'une année ou d'un mois à partir d'une date...).
- **Sélection de variables** : ne garder que les variables utiles et retirer celles qui sont inutiles ou redondantes, pour limiter le surapprentissage (overfitting).

### 4. Normalisation et mise à l'échelle (Scaling)

Nécessaire pour les algorithmes sensibles aux distances ou aux magnitudes (réseaux de neurones, k plus proches voisins). Les modèles à base d'arbres (Random Forest, XGBoost) n'en ont en général pas besoin.

- **Min-Max Scaling** : ramène les valeurs entre 0 et 1.
- **Standardisation** : centre les données autour d'une moyenne de 0 avec un écart-type de 1.

### 5. Division des données et vérification

Séparer `train` en deux parties pour évaluer le modèle sur des données qu'il n'a jamais vues :

- **Ensemble d'entraînement** (70 à 80 %) : le modèle y apprend les motifs.
- **Ensemble de validation** (20 à 30 %) : gardé de côté pendant l'apprentissage, pour mesurer objectivement la performance.

> Cet ensemble de validation est tiré de `train`. Il ne faut pas le confondre avec `test.csv`, dont la réponse est gardée secrète par Kaggle.

### 6. Entraînement

Entraîner le modèle (par exemple `RandomForestClassifier`), puis mesurer son score sur l'ensemble de validation.

---

## Phase II : table `test`

**But** : obtenir une prédiction pour chaque champignon de `test`.

- **Nettoyer sans jamais supprimer de lignes** : Kaggle attend une prédiction pour chaque ligne. Utiliser `fillna` et `clip` avec les valeurs calculées sur `train`.
- **Transformer comme `train`** : mêmes encodages, même mise à l'échelle, mêmes colonnes dans le même ordre.
- **Prédire** avec le modèle déjà entraîné (sans le ré-entraîner) :

```python
pred = model.predict(X_test)
```

---

## Phase III : soumission

**But** : produire le fichier à envoyer à Kaggle.

- Mettre les prédictions au format de `sample_submission.csv` : mêmes colonnes, mêmes noms, même nombre de lignes.
- Enregistrer et envoyer :

```python
submission = sample.copy()
submission.iloc[:, 1] = pred
submission.to_csv("submission.csv", index=False)
```

- Envoyer `submission.csv` via **Submit Predictions** sur la page de la compétition.

---

## Résumé

| Phase | Table | Actions |
|---|---|---|
| I | `train` | nettoyer, encoder, (créer / sélectionner des variables), mettre à l'échelle, séparer entraînement / validation, entraîner, vérifier |
| II | `test` | nettoyer **sans supprimer de lignes**, transformer comme `train`, prédire |
| III | `submission.csv` | mettre au format de `sample_submission`, envoyer |
