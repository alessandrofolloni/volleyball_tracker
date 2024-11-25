import re
import pandas as pd
from pathlib import Path

# Directory di base (puoi usare un percorso assoluto se necessario)
base_dir = Path('volleyball_tracker_training')  # Modifica questo percorso se necessario

# Verifica che la directory di base esista
if not base_dir.exists():
    print(f"La directory di base '{base_dir}' non esiste. Controlla il percorso.")
    exit()

# Pattern per trovare i file che iniziano con 'metrics_report_' e finiscono con '.txt'
report_prefix = 'metrics_report_'
report_suffix = '.txt'

# Lista per raccogliere i dati
data = []

# Set per raccogliere tutti i nomi delle metriche
all_metrics = set()


# Funzione per estrarre le metriche da un report
def extract_metrics(file_path):
    metrics = {}
    try:
        with file_path.open('r', encoding='utf-8') as file:
            for line in file:
                # Rimuovi eventuali spazi bianchi all'inizio e alla fine
                line = line.strip()

                # Estrazione del nome del modello
                if line.startswith('- Model name:'):
                    model_match = re.search(r'- Model name:\s*(.+)', line)
                    if model_match:
                        metrics['Model Name'] = model_match.group(1).strip()

                # Estrazione di Precision
                elif line.startswith('- Precision (P):'):
                    precision_match = re.search(r'- Precision \(P\):\s*([0-9.]+)', line)
                    if precision_match:
                        metrics['Precision'] = float(precision_match.group(1))

                # Estrazione di Recall
                elif line.startswith('- Recall (R):'):
                    recall_match = re.search(r'- Recall \(R\):\s*([0-9.]+)', line)
                    if recall_match:
                        metrics['Recall'] = float(recall_match.group(1))

                # Estrazione di mAP@0.5
                elif 'mAP@IoU=0.5 (mAP@0.5):' in line:
                    map_05_match = re.search(r'mAP@IoU=0\.5\s*\(mAP@0\.5\):\s*([0-9.]+)', line)
                    if map_05_match:
                        metrics['mAP@0.5'] = float(map_05_match.group(1))

                # Estrazione di mAP@0.5:0.95
                elif 'mAP@IoU=0.5:0.95 (mAP@0.5:0.95):' in line:
                    map_095_match = re.search(r'mAP@IoU=0\.5:0\.95\s*\(mAP@0\.5:0\.95\):\s*([0-9.]+)', line)
                    if map_095_match:
                        metrics['mAP@0.5:0.95'] = float(map_095_match.group(1))

                # Estrazione di Average Box Loss
                elif line.startswith('- Average Box Loss:'):
                    box_loss_match = re.search(r'- Average Box Loss:\s*([0-9.]+)', line)
                    if box_loss_match:
                        metrics['Average Box Loss'] = float(box_loss_match.group(1))

                # Estrazione di Average Classification Loss
                elif line.startswith('- Average Classification Loss:'):
                    class_loss_match = re.search(r'- Average Classification Loss:\s*([0-9.]+)', line)
                    if class_loss_match:
                        metrics['Average Classification Loss'] = float(class_loss_match.group(1))

                # Estrazione di Average DFL Loss
                elif line.startswith('- Average DFL Loss:'):
                    dfl_loss_match = re.search(r'- Average DFL Loss:\s*([0-9.]+)', line)
                    if dfl_loss_match:
                        metrics['Average DFL Loss'] = float(dfl_loss_match.group(1))

                # Puoi aggiungere ulteriori metriche qui se necessario

    except Exception as e:
        print(f"Errore nella lettura del file {file_path}: {e}")
        return None

    return metrics if metrics else None


# Naviga attraverso tutte le sottocartelle e raccoglie i report
print(f"Inizio la ricerca dei file in '{base_dir}'...")
for file_path in base_dir.rglob(f'{report_prefix}*{report_suffix}'):
    print(f"Trovato file: {file_path}")
    metrics = extract_metrics(file_path)
    if metrics:
        metrics['Path'] = str(file_path.parent)  # Aggiungi il percorso per riferimento
        data.append(metrics)
        all_metrics.update(metrics.keys())
        print(f"Metrice estratte per {file_path}: {metrics}")
    else:
        print(f"Nessuna metrica estratta da {file_path}. Salto questo file.")

if not data:
    print("Nessun file corrispondente trovato o metriche mancanti nei file trovati.")
    exit()

# Identifica tutte le metriche presenti
all_metrics.discard('Path')  # 'Path' non è una metrica
all_metrics.discard('Model Name')  # 'Model Name' non è una metrica

# Crea un DataFrame
df = pd.DataFrame(data)

# Riordina le colonne: Model Name, tutte le metriche in ordine, Path
ordered_columns = ['Model Name'] + sorted(all_metrics) + ['Path']
df = df.reindex(columns=ordered_columns)

# Identifica le migliori metriche per ciascuna colonna numerica
best_metrics = {}
for metric in sorted(all_metrics):
    if pd.api.types.is_numeric_dtype(df[metric]):
        best_metrics[metric] = df[metric].max()

print(f"Migliori metriche identificate: {best_metrics}")


# Funzione per formattare le metriche, evidenziando le migliori
def format_metric(value, metric_name):
    if pd.isna(value):
        return 'N/A'
    if metric_name in best_metrics and value == best_metrics[metric_name]:
        return f"**{value}**"
    else:
        return f"{value}"


# Crea il file riassuntivo in Markdown
summary_file = Path('summary_metrics.md')

with summary_file.open('w', encoding='utf-8') as f:
    # Scrivi l'intestazione
    header = '| ' + ' | '.join(ordered_columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(ordered_columns)) + ' |'
    f.write(header + '\n')
    f.write(separator + '\n')

    # Scrivi ogni riga con le metriche formattate
    for _, row in df.iterrows():
        row_data = []
        for col in ordered_columns:
            if col in ['Model Name', 'Path']:
                row_data.append(str(row[col]))
            else:
                row_data.append(format_metric(row[col], col))
        f.write('| ' + ' | '.join(row_data) + ' |\n')

print(f"File riassuntivo creato: {summary_file.resolve()}")