from pathlib import Path
import urllib.request
import urllib.error

DEPARTEMENTS = ["04", "05", "06", "13", "83", "84"]  # PACA
ANNEES = range(2019, 2027)  # on tente large, le script garde ce qui existe
BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"

dossier = Path("data/raw")
dossier.mkdir(parents=True, exist_ok=True)

telecharges, absents = [], []

for annee in ANNEES:
    for dep in DEPARTEMENTS:
        fichier = dossier / f"dvf_{annee}_{dep}.csv.gz"
        if fichier.exists():
            telecharges.append(fichier.name)
            continue  # déjà téléchargé
        try:
            print(f"Téléchargement {annee} - {dep}...")
            urllib.request.urlretrieve(BASE_URL.format(annee=annee, dep=dep), fichier)
            telecharges.append(fichier.name)
        except urllib.error.HTTPError as e:
            print(f"  -> indisponible ({e.code}), on passe")
            absents.append(f"{annee}-{dep}")

print(f"\nTerminé ! {len(telecharges)} fichiers présents dans {dossier}")
if absents:
    print(f"Non disponibles : {', '.join(absents)}")