
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")



mat = pd.read_csv(BASE_DIR / "student-mat.csv", sep=";")
por = pd.read_csv(BASE_DIR / "student-por.csv", sep=";")

mat["subject"] = "math"
por["subject"] = "portuguese"

df = pd.concat([mat, por], ignore_index=True)

print("Mathématiques :", mat.shape)
print("Portugais      :", por.shape)
print("Dataset total  :", df.shape)


dataset_path = OUTPUT_DIR / "student-all-combined.csv"

df.to_csv(dataset_path, index=False, encoding="utf-8")

print(f"\nDataset combiné enregistré dans : {dataset_path}")


df.info()

missing = df.isna().sum()
print(missing[missing > 0])
print("Total des valeurs manquantes :", df.isna().sum().sum())

print("Nombre de doublons :", df.duplicated().sum())

print(df.describe())

print(df.select_dtypes(include="object").nunique())




plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="G3",
    hue="subject",
    bins=21,
    multiple="layer",
    kde=False
)

plt.title("Distribution de la note finale selon la matière")
plt.xlabel("Note finale G3")
plt.ylabel("Nombre d'étudiants")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "distribution_G3.png", dpi=300)
plt.show()




df["pass"] = (df["G3"] >= 10).astype(int)

print(df["pass"].value_counts().rename(index={
    0: "Echec",
    1: "Reussite"
}))

print("\nProportions :")
print(df["pass"].value_counts(normalize=True).rename(index={
    0: "Echec",
    1: "Reussite"
}).round(3))




df["level"] = pd.cut(
    df["G3"],
    bins=[-1, 9, 13, 20],
    labels=["faible", "moyen", "bon"]
)

print(df["level"].value_counts().sort_index())

print("\nProportions :")
print(df["level"].value_counts(normalize=True).sort_index().round(3))



num = df.select_dtypes(include="number").drop(
    columns=["pass"],
    errors="ignore"
)

correlation = num.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7}
)

plt.title("Matrice de corrélation des variables numériques")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=300)
plt.show()



keys = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery",
    "internet"
]

df["student_id"] = df.groupby(keys, dropna=False).ngroup()

print("Nombre de profils distincts :", df["student_id"].nunique())

repeated_profiles = (
    df.groupby("student_id").size().gt(1).sum()
)

print("Profils présents plusieurs fois :", repeated_profiles)




report_path = OUTPUT_DIR / "eda_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("RAPPORT D'ANALYSE EXPLORATOIRE (EDA)\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Dimensions mathématiques : {mat.shape}\n")
    file.write(f"Dimensions portugais : {por.shape}\n")
    file.write(f"Dimensions dataset combiné : {df.shape}\n\n")

    file.write("Valeurs manquantes par colonne :\n")
    file.write(df.isna().sum().to_string() + "\n\n")

    file.write(f"Nombre de doublons exacts : {df.duplicated().sum()}\n\n")

    file.write("Statistiques descriptives :\n")
    file.write(df.describe().to_string() + "\n\n")

    file.write("Répartition réussite / échec :\n")
    file.write(df["pass"].value_counts(normalize=True).to_string() + "\n\n")

    file.write("Répartition des niveaux :\n")
    file.write(df["level"].value_counts(normalize=True).to_string() + "\n\n")

    file.write("Corrélations avec G3 :\n")
    file.write(
        correlation["G3"].sort_values(ascending=False).to_string()
        + "\n"
    )

print("Graphiques et rapports enregistrés dans :", OUTPUT_DIR)
print("Rapport descriptif :", report_path)
# Vérifier les dimensions
print("Dimensions :", df.shape)

# Vérifier les types
print(df.dtypes)

# Vérifier les valeurs manquantes
print(df.isnull().sum().sort_values(ascending=False))

# Vérifier les doublons exacts
print("Doublons :", df.duplicated().sum())

# Vérifier les valeurs uniques
print(df.nunique().sort_values())

# Vérifier les notes
for col in ["G1", "G2", "G3"]:
    print(f"\nDistribution de {col}")
    print(df[col].describe())
    print("Valeurs hors [0, 20] :",
          ((df[col] < 0) | (df[col] > 20)).sum())



plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="subject",
    y="G3"
)

plt.title("Comparaison des notes finales par matière")
plt.xlabel("Matière")
plt.ylabel("Note finale G3")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "boxplot_G3_subject.png", dpi=300)
plt.show()



success_by_subject = (
    df.groupby("subject", observed=True)["pass"]
      .mean()
      .mul(100)
      .reset_index(name="success_rate")
)

print(success_by_subject)

plt.figure(figsize=(7, 5))

sns.barplot(
    data=success_by_subject,
    x="subject",
    y="success_rate"
)

plt.title("Taux de réussite selon la matière")
plt.xlabel("Matière")
plt.ylabel("Taux de réussite (%)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "success_rate_subject.png", dpi=300)
plt.show()


num = (
    df.select_dtypes(include="number")
      .drop(columns=["pass"], errors="ignore")
)

correlation = num.corr()

print(
    correlation["G3"]
    .sort_values(ascending=False)
)





BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "reports" / "student-all-combined.csv"

df = pd.read_csv(DATA_PATH)

# Recréer la cible si nécessaire
df["pass"] = (df["G3"] >= 10).astype(int)

# Colonnes interdites comme variables explicatives
excluded = ["G3", "pass", "level", "student_id"]

X = df.drop(columns=excluded, errors="ignore")
y = df["pass"]

# Identifier les colonnes numériques et catégorielles
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

# Préparation des variables numériques
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Préparation des variables catégorielles
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combiner les deux préparations
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Séparer les données avant l'apprentissage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Taille de X :", X.shape)
print("Taille de l'entraînement :", X_train.shape)
print("Taille du test :", X_test.shape)
print("\nRépartition de la cible :")
print(y.value_counts(normalize=True).round(3))



models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", clone(model))
    ])

    # Validation croisée uniquement sur les données d'entraînement
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "balanced_accuracy": "balanced_accuracy"
        },
        n_jobs=-1
    )

    # Entraînement final sur l'ensemble d'entraînement
    pipeline.fit(X_train, y_train)

    # Évaluation sur le test indépendant
    y_pred = pipeline.predict(X_test)

    results.append({
        "model": name,
        "cv_f1_mean": cv_scores["test_f1"].mean(),
        "cv_f1_std": cv_scores["test_f1"].std(),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, y_pred
        )
    })

    trained_models[name] = pipeline

    print(f"\n{name} ")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Echec", "Reussite"],
        zero_division=0
    ))

results_df = pd.DataFrame(results).sort_values(
    "cv_f1_mean",
    ascending=False
)

print("\nCOMPARAISON DES MODELES")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(
    BASE_DIR / "reports" / "classification_results.csv",
    index=False
)





OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Graphique de comparaison des F1-scores
plot_data = results_df.sort_values("f1", ascending=False)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=plot_data,
    x="f1",
    y="model"
)

plt.xlim(0, 1)
plt.title("Comparaison des modèles de classification")
plt.xlabel("F1-score sur le jeu de test")
plt.ylabel("Modèle")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "classification_comparison.png",
    dpi=300
)
plt.show()



for name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Echec", "Reussite"],
        yticklabels=["Echec", "Reussite"]
    )

    plt.title(f"Matrice de confusion - {name}")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_DIR / f"confusion_matrix_{safe_name}.png",
        dpi=300
    )

    plt.show()




rf_pipeline = clone(trained_models["Random Forest"])

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_search.best_params_)
print("Meilleur F1 en validation croisée :",
      round(grid_search.best_score_, 3))

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

print(classification_report(
    y_test,
    y_pred_best,
    target_names=["Echec", "Reussite"],
    zero_division=0
))



excluded_reg = [
    "G3", "pass", "level", "student_id"
]

X_reg = df.drop(
    columns=excluded_reg,
    errors="ignore"
)

y_reg = df["G3"]

numeric_reg = X_reg.select_dtypes(
    include="number"
).columns.tolist()

categorical_reg = X_reg.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_reg_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_reg_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_reg = ColumnTransformer([
    ("num", numeric_reg_pipeline, numeric_reg),
    ("cat", categorical_reg_pipeline, categorical_reg)
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

regression_models = {
    "Baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}

regression_results = []
regression_pipelines = {}

for name, model in regression_models.items():

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor_reg)),
        ("model", clone(model))
    ])

    pipeline.fit(X_train_reg, y_train_reg)

    y_pred_reg = pipeline.predict(X_test_reg)

    mae = mean_absolute_error(
        y_test_reg, y_pred_reg
    )

    rmse = np.sqrt(
        mean_squared_error(y_test_reg, y_pred_reg)
    )

    r2 = r2_score(y_test_reg, y_pred_reg)

    regression_results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    regression_pipelines[name] = pipeline

regression_results_df = pd.DataFrame(
    regression_results
).sort_values("MAE")

print("\n========== RESULTATS DE REGRESSION ==========")
print(regression_results_df.round(3).to_string(index=False))

regression_results_df.to_csv(
    OUTPUT_DIR / "regression_results.csv",
    index=False
)


best_reg_name = regression_results_df.iloc[0]["model"]
best_reg_model = regression_pipelines[best_reg_name]

predictions = best_reg_model.predict(X_test_reg)

plt.figure(figsize=(7, 6))

sns.scatterplot(
    x=y_test_reg,
    y=predictions
)

plt.plot([0, 20], [0, 20], linestyle="--")

plt.xlim(0, 20)
plt.ylim(0, 20)

plt.xlabel("Notes réelles")
plt.ylabel("Notes prédites")
plt.title(f"Notes réelles vs prédites - {best_reg_name}")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "regression_predictions.png",
    dpi=300
)
plt.show()


excluded = [
    "G3", "pass", "level", "student_id",
    "G1", "G2"
]

X = df.drop(columns=excluded, errors="ignore")
y = df["pass"]




best_name = results_df.iloc[0]["model"]
best_pipeline = trained_models[best_name]

importance = permutation_importance(
    best_pipeline,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
}).sort_values(
    "importance_mean",
    ascending=False
)

print("\n========== IMPORTANCE DES VARIABLES ==========")
print(importance_df.head(15).round(4).to_string(index=False))

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

top_features = importance_df.head(15).sort_values(
    "importance_mean"
)

plt.figure(figsize=(9, 7))

plt.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"]
)

plt.xlabel("Diminution moyenne du F1-score après permutation")
plt.ylabel("Variable")
plt.title("Importance des variables - Permutation")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)
plt.show()