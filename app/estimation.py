"""Calcul des variables et géocodage pour l'application d'estimation.

Les calculs reprennent exactement ceux du notebook 04_features.ipynb, pour que le modèle
reçoive des variables construites de la même façon qu'à l'entraînement.
"""
from datetime import date

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

RAYON_TERRE_KM = 6371.0
K_VOISINES = 20
DEPARTEMENTS_PACA = {"04", "05", "06", "13", "83", "84"}

FEATURES = [
    "est_maison", "log_surface", "nombre_pieces_principales", "surface_par_piece", "log_terrain", "nb_dependances",
    "mois_depuis_2021", "mois",
    "latitude", "longitude", "dist_mer_km", "dist_grande_ville_km",
    "dist_marseille_km", "dist_nice_km", "dist_toulon_km", "dist_aix_km", "dist_avignon_km",
    "voisins_log_prix_m2", "voisins_distance_km",
]

VILLES = {
    "marseille": (43.2965, 5.3698),
    "nice": (43.7102, 7.2620),
    "toulon": (43.1242, 5.9280),
    "aix": (43.5297, 5.4474),
    "avignon": (43.9493, 4.8055),
}

# Points du littoral (latitude, longitude), d'ouest en est — identiques au notebook 04
LITTORAL = [
    (43.452, 4.428), (43.360, 4.580), (43.355, 4.730), (43.375, 4.830), (43.400, 4.950),
    (43.330, 5.040), (43.330, 5.110), (43.330, 5.150), (43.340, 5.250), (43.360, 5.310),
    (43.295, 5.365), (43.260, 5.375), (43.245, 5.370), (43.215, 5.350), (43.210, 5.420),
    (43.210, 5.445), (43.205, 5.500), (43.213, 5.535), (43.200, 5.560), (43.175, 5.607),
    (43.175, 5.680), (43.135, 5.750), (43.117, 5.800), (43.075, 5.800), (43.050, 5.860),
    (43.090, 5.890), (43.115, 5.930), (43.105, 5.940), (43.095, 5.960), (43.085, 6.020),
    (43.085, 6.070), (43.070, 6.130), (43.030, 6.130), (43.090, 6.160), (43.110, 6.170),
    (43.120, 6.200), (43.125, 6.240), (43.100, 6.320), (43.137, 6.370), (43.150, 6.430),
    (43.160, 6.480), (43.170, 6.530), (43.165, 6.620), (43.200, 6.680), (43.230, 6.660),
    (43.270, 6.640), (43.270, 6.585), (43.310, 6.640), (43.340, 6.720), (43.390, 6.730),
    (43.420, 6.740), (43.420, 6.770), (43.430, 6.860), (43.510, 6.940), (43.520, 6.940),
    (43.550, 7.010), (43.565, 7.070), (43.565, 7.110), (43.545, 7.130), (43.585, 7.130),
    (43.640, 7.140), (43.655, 7.160), (43.665, 7.190), (43.660, 7.210), (43.690, 7.250),
    (43.695, 7.285), (43.700, 7.310), (43.685, 7.330), (43.705, 7.335), (43.715, 7.360),
    (43.720, 7.400), (43.735, 7.425), (43.755, 7.470), (43.775, 7.500), (43.785, 7.530),
]


def distance_km(lat1, lon1, lat2, lon2):
    """Distance à vol d'oiseau (formule de haversine), en km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RAYON_TERRE_KM * np.arcsin(np.sqrt(a))


def densifier(points, pas_km=0.2):
    """Ajoute des points intermédiaires tous les `pas_km` km le long de la ligne."""
    resultat = []
    for (lat1, lon1), (lat2, lon2) in zip(points[:-1], points[1:]):
        n = max(1, int(np.ceil(distance_km(lat1, lon1, lat2, lon2) / pas_km)))
        t = np.linspace(0, 1, n, endpoint=False)
        resultat.append(np.column_stack([lat1 + t * (lat2 - lat1), lon1 + t * (lon2 - lon1)]))
    resultat.append(np.array([points[-1]]))
    return np.vstack(resultat)


class Referentiel:
    """Le littoral et les ventes passées, indexés pour retrouver rapidement les plus proches."""

    def __init__(self, ventes: pd.DataFrame):
        self.arbre_littoral = BallTree(np.radians(densifier(LITTORAL)), metric="haversine")
        self.ventes = {}
        for est_maison in (0, 1):
            sous = ventes[ventes["est_maison"] == est_maison].reset_index(drop=True)
            coords = np.radians(sous[["latitude", "longitude"]].to_numpy(dtype=float))
            self.ventes[est_maison] = (BallTree(coords, metric="haversine"), sous)

    def distance_mer_km(self, lat, lon):
        distance, _ = self.arbre_littoral.query(np.radians([[lat, lon]]), k=1)
        return float(distance[0, 0] * RAYON_TERRE_KM)

    def voisines(self, lat, lon, est_maison, k=K_VOISINES):
        """Les k ventes passées du même type les plus proches, avec leur distance en km."""
        arbre, sous = self.ventes[est_maison]
        distance, index = arbre.query(np.radians([[lat, lon]]), k=k)
        resultat = sous.iloc[index[0]].copy()
        resultat["distance_km"] = distance[0] * RAYON_TERRE_KM
        return resultat


def calculer_features(referentiel, lat, lon, est_maison, surface, pieces, terrain, dependances, jour=None):
    """Construit la ligne de variables attendue par le modèle, et renvoie aussi les ventes voisines."""
    jour = jour or date.today()
    voisines = referentiel.voisines(lat, lon, est_maison)
    ligne = {
        "est_maison": est_maison,
        "log_surface": np.log(surface),
        "nombre_pieces_principales": pieces,
        "surface_par_piece": surface / pieces,
        "log_terrain": np.log1p(terrain if est_maison else 0),
        "nb_dependances": dependances,
        "mois_depuis_2021": (jour.year - 2021) * 12 + jour.month - 1,
        "mois": jour.month,
        "latitude": lat,
        "longitude": lon,
        "dist_mer_km": referentiel.distance_mer_km(lat, lon),
        "voisins_log_prix_m2": float(np.median(voisines["log_prix_m2"])),
        "voisins_distance_km": float(voisines["distance_km"].iloc[-1]),
    }
    for ville, (lat_v, lon_v) in VILLES.items():
        ligne[f"dist_{ville}_km"] = float(distance_km(lat, lon, lat_v, lon_v))
    ligne["dist_grande_ville_km"] = min(ligne[f"dist_{ville}_km"] for ville in VILLES)
    return pd.DataFrame([ligne])[FEATURES], voisines


# Base Adresse Nationale : service officiel et gratuit de géocodage des adresses françaises.
# L'API historique (api-adresse.data.gouv.fr) a été reprise par la Géoplateforme de l'IGN :
# on essaie la nouvelle adresse, puis l'ancienne en secours.
SERVICES_GEOCODAGE = [
    "https://data.geopf.fr/geocodage/search",
    "https://api-adresse.data.gouv.fr/search/",
]


def geocoder(adresse):
    """Renvoie {label, latitude, longitude, code_commune} pour une adresse, ou None si introuvable."""
    for url in SERVICES_GEOCODAGE:
        try:
            reponse = requests.get(url, params={"q": adresse, "limit": 1}, timeout=8)
            reponse.raise_for_status()
            resultats = reponse.json().get("features", [])
        except (requests.RequestException, ValueError):
            continue
        if not resultats:
            return None
        lon, lat = resultats[0]["geometry"]["coordinates"]
        proprietes = resultats[0]["properties"]
        return {
            "label": proprietes.get("label", adresse),
            "latitude": lat,
            "longitude": lon,
            "code_commune": proprietes.get("citycode", ""),
            "score": proprietes.get("score", 1.0),
        }
    raise ConnectionError("Le service de géocodage ne répond pas")


def est_en_paca(code_commune):
    return str(code_commune)[:2] in DEPARTEMENTS_PACA
