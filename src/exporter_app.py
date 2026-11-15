"""Prépare les fichiers dont l'application Streamlit a besoin.

À lancer depuis la racine du projet, après les notebooks 04 et 05 :

    python src/exporter_app.py

Crée dans app/assets/ :
- modele_lightgbm.txt : le modèle LightGBM au format texte (lisible quelle que soit la version de LightGBM)
- reference_ventes.parquet : la position et le prix au m² de toutes les ventes, pour calculer le prix des voisines
"""
from pathlib import Path

import joblib
import pandas as pd

racine = Path(__file__).resolve().parent.parent
dossier_assets = racine / "app" / "assets"
dossier_assets.mkdir(parents=True, exist_ok=True)

# 1. Le modèle
chemin_modele = racine / "models" / "lightgbm.joblib"
if not chemin_modele.exists():
    raise SystemExit(f"Modèle introuvable : {chemin_modele}\n→ Lance d'abord le notebook 05_modelisation.")
sauvegarde = joblib.load(chemin_modele)
booster = sauvegarde["modele"].booster_
assert booster.feature_name() == sauvegarde["features"], "Les noms de variables du modèle ne correspondent pas"
booster.save_model(str(dossier_assets / "modele_lightgbm.txt"))

# 2. Les ventes de référence (toutes les années : pour une estimation aujourd'hui, tout le passé est utilisable)
features = pd.read_parquet(racine / "data" / "processed" / "features.parquet")
reference = pd.DataFrame({
    "latitude": features["latitude"].astype("float32"),
    "longitude": features["longitude"].astype("float32"),
    "log_prix_m2": features["log_prix_m2"].astype("float32"),
    "est_maison": features["est_maison"].astype("int8"),
}).reset_index(drop=True)
reference.to_parquet(dossier_assets / "reference_ventes.parquet", index=False)

for fichier in sorted(dossier_assets.iterdir()):
    print(f"{fichier.relative_to(racine)} : {fichier.stat().st_size / 1e6:.1f} Mo")
print(f"{len(reference):,} ventes de référence, {booster.num_trees()} arbres")
