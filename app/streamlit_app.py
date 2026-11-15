"""Application d'estimation du prix d'un logement en PACA.

Lancement en local, depuis la racine du projet :
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimation  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
URL_GITHUB = "https://github.com/salwahmch/estimateur-immo-dvf"

# Erreur médiane du modèle sur les ventes de 2025 (notebook 05), par type de bien
ERREUR_MEDIANE = {0: 0.142, 1: 0.166}

GROUPES = {
    "Prix du quartier": ["voisins_log_prix_m2", "voisins_distance_km"],
    "Surface": ["log_surface", "surface_par_piece"],
    "Nombre de pièces": ["nombre_pieces_principales"],
    "Terrain": ["log_terrain"],
    "Dépendances (cave, garage…)": ["nb_dependances"],
    "Distance à la mer": ["dist_mer_km"],
    "Situation dans la région": ["latitude", "longitude", "dist_grande_ville_km", "dist_marseille_km",
                                 "dist_nice_km", "dist_toulon_km", "dist_aix_km", "dist_avignon_km"],
    "Maison ou appartement": ["est_maison"],
    "Date de l'estimation": ["mois_depuis_2021", "mois"],
}


def euros(montant):
    return f"{montant:,.0f} €".replace(",", " ")


@st.cache_resource
def charger():
    modele = lgb.Booster(model_file=str(ASSETS / "modele_lightgbm.txt"))
    assert modele.feature_name() == estimation.FEATURES, "Le modèle ne correspond pas aux variables de l'application"
    referentiel = estimation.Referentiel(pd.read_parquet(ASSETS / "reference_ventes.parquet"))
    return modele, referentiel


st.set_page_config(page_title="Estimateur immobilier PACA", page_icon="🏠", layout="centered")
st.title("🏠 Estimateur immobilier PACA")
st.write(
    "Estimez le prix d'un appartement ou d'une maison en Provence-Alpes-Côte d'Azur. "
    "Le modèle a appris sur **434 000 ventes réelles** enregistrées entre 2021 et 2025 (données DVF)."
)

modele, referentiel = charger()

with st.form("bien"):
    adresse = st.text_input("Adresse du bien", placeholder="Ex. : 12 rue Paradis, Marseille")
    type_bien = st.radio("Type de bien", ["Appartement", "Maison"], horizontal=True)
    colonne_1, colonne_2 = st.columns(2)
    surface = colonne_1.number_input("Surface habitable (m²)", min_value=9, max_value=1000, value=65)
    pieces = colonne_2.number_input("Nombre de pièces principales", min_value=1, max_value=20, value=3)
    terrain = colonne_1.number_input("Surface du terrain (m², maisons)", min_value=0, max_value=100_000, value=0,
                                     help="Ignorée pour un appartement")
    dependances = colonne_2.number_input("Dépendances (cave, garage, parking)", min_value=0, max_value=10, value=0)
    valider = st.form_submit_button("Estimer le prix", type="primary")

if valider:
    if not adresse.strip():
        st.warning("Indiquez une adresse.")
        st.stop()

    try:
        lieu = estimation.geocoder(adresse)
    except ConnectionError:
        st.error("Le service de recherche d'adresses ne répond pas. Réessayez dans quelques instants.")
        st.stop()
    if lieu is None:
        st.error("Adresse introuvable. Vérifiez l'orthographe ou ajoutez la commune.")
        st.stop()
    if not estimation.est_en_paca(lieu["code_commune"]):
        st.error(f"« {lieu['label']} » n'est pas en région PACA. Le modèle ne couvre que les départements "
                 "04, 05, 06, 13, 83 et 84.")
        st.stop()

    est_maison = int(type_bien == "Maison")
    X, voisines = estimation.calculer_features(
        referentiel, lieu["latitude"], lieu["longitude"], est_maison, surface, pieces, terrain, dependances,
    )
    log_prix = modele.predict(X)[0]
    prix = float(np.exp(log_prix))
    erreur = ERREUR_MEDIANE[est_maison]

    st.subheader(lieu["label"])
    if lieu["score"] < 0.5:
        st.warning("L'adresse trouvée ne correspond peut-être pas exactement à celle saisie : vérifiez-la ci-dessus.")

    a, b, c = st.columns(3)
    a.metric("Prix estimé", euros(prix))
    b.metric("Prix au m²", euros(prix / surface))
    c.metric("Fourchette indicative", f"± {erreur * 100:.0f} %")
    st.caption(
        f"Fourchette : **{euros(prix * (1 - erreur))} – {euros(prix * (1 + erreur))}**. Sur les ventes de 2025, "
        f"la moitié des estimations du modèle pour ce type de bien se trompaient de moins de {erreur * 100:.0f} %."
    )
    if X["voisins_distance_km"].iloc[0] > 5:
        st.warning("Peu de ventes ont eu lieu près de cette adresse : l'estimation est moins fiable.")

    # Les facteurs qui expliquent l'estimation (valeurs SHAP calculées par LightGBM)
    st.markdown("#### Pourquoi ce prix ?")
    contributions = modele.predict(X, pred_contrib=True)[0]
    prix_moyen = float(np.exp(contributions[-1]))
    par_variable = pd.Series(contributions[:-1], index=estimation.FEATURES)
    effets = pd.Series({nom: par_variable[cols].sum() for nom, cols in GROUPES.items()})
    effets = effets[effets.abs() > 0.005].sort_values()

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(effets) + 1))
    pourcentages = (np.exp(effets) - 1) * 100
    barres = ax.barh(effets.index, pourcentages, color=["#c0392b" if p < 0 else "#27ae60" for p in pourcentages])
    ax.bar_label(barres, labels=[f"{p:+.0f} %" for p in pourcentages], padding=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Effet sur le prix estimé")
    ax.spines[["top", "right"]].set_visible(False)
    marge = max(10, pourcentages.abs().max() * 1.3)
    ax.set_xlim(-marge if (pourcentages < 0).any() else 0, marge)
    st.pyplot(fig)
    st.caption(
        f"Point de départ : le prix d'un logement « moyen » selon le modèle, soit {euros(prix_moyen)}. "
        "Chaque barre indique de combien un facteur fait monter (vert) ou baisser (rouge) l'estimation. "
        "Les effets se multiplient entre eux."
    )

    # Le quartier
    st.markdown("#### Les 20 ventes comparables les plus proches")
    prix_m2_voisines = np.exp(voisines["log_prix_m2"])
    rayon_m = euros(voisines["distance_km"].iloc[-1] * 1000).replace("€", "m")
    st.write(f"Prix au m² médian : **{euros(prix_m2_voisines.median())}**, dans un rayon de **{rayon_m}**.")
    carte = pd.concat([
        pd.DataFrame({"lat": voisines["latitude"], "lon": voisines["longitude"], "couleur": "#2e86de", "taille": 15}),
        pd.DataFrame({"lat": [lieu["latitude"]], "lon": [lieu["longitude"]], "couleur": "#e74c3c", "taille": 30}),
    ])
    st.map(carte, latitude="lat", longitude="lon", color="couleur", size="taille")
    st.caption("🔴 Le bien estimé — 🔵 les ventes comparables")

with st.expander("Méthode et limites"):
    st.markdown(f"""
- **Données :** ventes de logements anciens en PACA de 2021 à 2025, issues des *Demandes de valeurs foncières* (DGFiP, open data).
- **Modèle :** LightGBM, entraîné sur les ventes 2021-2024 et évalué sur celles de 2025 : **erreur médiane de 14,9 %**.
- **Variables :** surface, pièces, terrain, dépendances, distance à la mer et aux grandes villes, et surtout le prix au m² des 20 ventes comparables les plus proches.
- **Limites :** le modèle ignore l'état du bien, l'étage, la vue, la présence d'une piscine ou la performance énergétique, qui ne figurent pas dans les données. Il ne couvre ni le neuf, ni l'immobilier de grand luxe.

Projet de data science : [code source et analyse complète sur GitHub]({URL_GITHUB}).
""")
