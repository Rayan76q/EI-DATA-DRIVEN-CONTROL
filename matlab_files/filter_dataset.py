import os
import sys
import time
from pathlib import Path

# Liste des sorties à extraire (numérotation de 1 à 18, comme dans MATLAB) :
#  1 : Position X                   10 : Angle Roll (Phi)
#  2 : Position Y                   11 : Angle Pitch (Theta)
#  3 : Position Z (Altitude)        12 : Angle Yaw (Psi)
#  4 : Vitesse Vx                   13 : Vitesse angulaire wx
#  5 : Vitesse Vy                   14 : Vitesse angulaire wy
#  6 : Vitesse Vz                   15 : Vitesse angulaire wz
#  7 : Accélération Ax              16 : Accélération angulaire wx_dot
#  8 : Accélération Ay              17 : Accélération angulaire wy_dot
#  9 : Accélération Az              18 : Accélération angulaire wz_dot
#
SELECTED_OUTPUTS = [7, 8, 9, 16, 17, 18]

# Noms des fichiers d'entrée et de sortie
SCRIPT_DIR = Path(__file__).resolve().parent

# A modifier selon la db à filtrer
INPUT_FILE = SCRIPT_DIR / "db_ext_struct_output.csv"
OUTPUT_FILE = SCRIPT_DIR / "db_ext_struct_output_filtered.csv"
DB_NAME = "Base Structurée (db_ext_struct_output)"


def filter_database(input_file, output_file, selected_outputs, total_outputs=18):
    """
    Filtre un fichier CSV où chaque ligne est de la forme :
    y1(t0), y2(t0), ..., y18(t0), y1(t1), y2(t1), ..., y18(t1), ...
    
    Pour ne conserver que les sorties spécifiées dans selected_outputs (1-indexé).
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"[ERREUR] Le fichier d'entrée n'existe pas : {input_path}")
        return False

    # Validation et conversion des indices (1-indexé vers 0-indexé)
    for out_num in selected_outputs:
        if not (1 <= out_num <= total_outputs):
            raise ValueError(f"Numéro de sortie invalide : {out_num}. Doit être compris entre 1 et {total_outputs}.")

    selected_indices_0based = [out_num - 1 for out_num in selected_outputs]
    n_selected = len(selected_indices_0based)

    print(f"\nTraitement de : {input_path.name}")
    print(f"Sorties sélectionnées ({n_selected}/{total_outputs}) : {selected_outputs}")
    print(f"Fichier de sortie : {output_path.name}")

    start_time = time.time()
    total_lines = 0
    cache_indices = {}

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:

        for line_idx, line in enumerate(f_in, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            # Suppression de la virgule finale éventuelle et découpage
            tokens = line_str.rstrip(',').split(',')
            n_tokens = len(tokens)

            if n_tokens % total_outputs != 0:
                print(f"[ATTENTION] Ligne {line_idx} : {n_tokens} valeurs (non divisible par {total_outputs}). Ligne ignorée.")
                continue

            n_steps = n_tokens // total_outputs

            # Mise en cache des indices précalculés selon le nombre de pas de temps
            if n_steps not in cache_indices:
                cache_indices[n_steps] = [
                    step * total_outputs + idx
                    for step in range(n_steps)
                    for idx in selected_indices_0based
                ]
            indices = cache_indices[n_steps]

            # Extraction et réassemblage au format CSV (avec virgule finale comme MATLAB)
            filtered_tokens = [tokens[i] for i in indices]
            f_out.write(','.join(filtered_tokens) + ',\n')

            total_lines += 1
            if line_idx % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  -> {line_idx} lignes traitées ({elapsed:.1f}s écoulées)...")

    elapsed = time.time() - start_time
    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCÈS] {total_lines} lignes traitées en {elapsed:.2f}s.")
    print(f"         Taille générée : {output_size_mb:.2f} Mo.")
    return True


def main():
    # Permet de passer des sorties en argument de ligne de commande si souhaité
    selected = SELECTED_OUTPUTS
    if len(sys.argv) > 1:
        try:
            selected = [int(arg) for arg in sys.argv[1:]]
            print(f"Sorties configurées via la ligne de commande : {selected}")
        except ValueError:
            print("[INFO] Arguments invalides, utilisation de la sélection par défaut dans le script.")

    print("=" * 70)
    print("EXTRACTION ET FILTRAGE DE LA BASE DE DONNÉES")
    print("=" * 70)
    print(f"Sorties choisies : {selected}")

    print("\n" + "-" * 70)
    print(f"-> {DB_NAME}")
    print("-" * 70)

    success = filter_database(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        selected_outputs=selected
    )

    if success:
        print("\n" + "=" * 70)
        print("La base filtrée a été générée avec succès !")
        print("=" * 70)


if __name__ == "__main__":
    main()