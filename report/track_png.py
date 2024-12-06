import cv2
import os
from pathlib import Path

# Configurazione delle directory
BASE_TRACK_DIR = "/public.hpc/alessandro.folloni2/volleyball_tracker/runs/track2"
OUTPUT_COMPARISON_DIR = "/public.hpc/alessandro.folloni2/volleyball_tracker/runs/track2/comparison_frames"

# Lista dei tracker (nomi delle sottocartelle)
TRACKERS = ["tracker_yolov8_default", "tracker_bytetrack", "tracker_botsort"]

# Lista dei timestamp in secondi da cui estrarre i frame
TIMESTAMPS = [10.0, 20.0, 30.0]  # Esempio: 10 secondi, 20 secondi, 30 secondi

def extract_frame(video_path, timestamp_sec):
    """
    Estrae un frame da un video a un timestamp specifico.

    Args:
        video_path (str): Percorso al file video.
        timestamp_sec (float): Timestamp in secondi.

    Returns:
        frame (numpy.ndarray): Frame estratto o None se non trovato.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return None

    # Calcola il numero di frame da saltare
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Impossibile determinare il FPS del video: {video_path}")
        cap.release()
        return None

    frame_number = int(round(timestamp_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Errore nell'estrazione del frame a {timestamp_sec} secondi dal video: {video_path}")
        cap.release()
        return None

    cap.release()
    return frame

def main():
    # Creazione della directory di output se non esiste
    os.makedirs(OUTPUT_COMPARISON_DIR, exist_ok=True)

    # Itera su tutte le cartelle di match nella directory base
    for match_folder in os.listdir(BASE_TRACK_DIR):
        match_path = os.path.join(BASE_TRACK_DIR, match_folder)
        if not os.path.isdir(match_path):
            continue  # Salta se non è una cartella

        print(f"Processing match: {match_folder}")
        # Creazione della directory di output per il match corrente
        match_output_dir = os.path.join(OUTPUT_COMPARISON_DIR, match_folder)
        os.makedirs(match_output_dir, exist_ok=True)

        # Verifica che tutte le cartelle dei tracker esistano per questo match
        trackers_present = all(os.path.isdir(os.path.join(match_path, tracker)) for tracker in TRACKERS)
        if not trackers_present:
            print(f"Alcuni tracker mancanti nella cartella {match_folder}. Saltando questo match.")
            continue

        # Verifica che ogni tracker abbia esattamente un video
        video_files = {}
        for tracker in TRACKERS:
            tracker_path = os.path.join(match_path, tracker)
            videos = [f for f in os.listdir(tracker_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if len(videos) != 1:
                print(f"Tracker {tracker} in {match_folder} ha {len(videos)} video. Deve averne esattamente uno. Saltando questo match.")
                video_files = {}
                break
            video_files[tracker] = os.path.join(tracker_path, videos[0])

        if not video_files:
            continue  # Se ci sono problemi con i video, passa al match successivo

        # Estrarre frame per ogni timestamp
        for ts in TIMESTAMPS:
            # Creazione di una sottocartella per ogni timestamp
            ts_folder_name = f"frame_t{ts:.2f}s"
            ts_output_dir = os.path.join(match_output_dir, ts_folder_name)
            os.makedirs(ts_output_dir, exist_ok=True)

            print(f"  Extracting frame at {ts} secondi.")

            for tracker in TRACKERS:
                video_path = video_files[tracker]
                frame = extract_frame(video_path, ts)
                if frame is not None:
                    # Nome del file immagine: trackername_frame_tXX.XXs.jpg
                    output_filename = f"{tracker}_frame_t{ts:.2f}s.jpg"
                    output_path = os.path.join(ts_output_dir, output_filename)
                    cv2.imwrite(output_path, frame)
                    print(f"    Salvato: {output_filename}")
                else:
                    print(f"    Frame a {ts} secondi non trovato per {tracker}.")

    print("\nEstrazione dei frame completata. Le immagini sono salvate in:", OUTPUT_COMPARISON_DIR)

if __name__ == "__main__":
    main()