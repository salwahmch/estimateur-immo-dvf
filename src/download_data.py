from pathlib import Path
import urllib.request

DEPARTEMENTS = ["04", "05", "06", "13", "83", "84"]  # PACA
ANNEES = [2020, 2021, 2022, 2023, 2024]  # à ajuster selon ce qui est dispo sur le site
BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"

dossier = Path("data/raw")
dossier.mkdir(parents=True, exist_ok=True)

for annee in ANNEES:
    for dep in DEPARTEMENTS:
        fichier = dossier / f"dvf_{annee}_{dep}.csv.gz"
        if fichier.exists():
            continue  # déjà téléchargé
        print(f"Téléchargement {annee} - {dep}...")
        urllib.request.urlretrieve(BASE_URL.format(annee=annee, dep=dep), fichier)

print("Terminé !")