# TTS-Lab

Lokale Web-UI, um Voice-Cloning-Modelle mit denselben Stimmproben und Texten zu vergleichen.
Gedacht für die Vertonung von Erklärvideos, nicht für Echtzeit.

Alles läuft auf dem eigenen Rechner. Nichts wird hochgeladen.

## Voraussetzungen

- Mac mit Apple Silicon (M1 oder neuer). Alle Engines laufen über Apple MLX.
- Python 3.11 und [uv](https://docs.astral.sh/uv/)
- Reichlich Plattenplatz: Die Modelle werden beim ersten Benutzen heruntergeladen (mehrere GB).

## Starten

```bash
uv sync
uv run python app.py
```

Dann http://127.0.0.1:7860 öffnen. In Safari, in Chrome hängt die Oberfläche beim Hochladen
von Stimmproben.

## Engines

| Engine | Modell | Lizenz | Besonderheit |
|---|---|---|---|
| Qwen3-TTS 1.7B | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | Apache 2.0 | Klonen ab 3 s Stimmprobe, 10 Sprachen |
| VoxCPM2 | `mlx-community/VoxCPM2-bf16` | Apache 2.0 | 30 Sprachen, nutzt das Transkript der Stimmprobe |
| Chatterbox Multilingual | `mlx-community/chatterbox-multilingual-v3` | MIT | Vergleichsbasis, 23 Sprachen, nur mit Stimmprobe |

Chatterbox hat in dieser MLX-Fassung keine eingebaute Standardstimme, es braucht also immer
eine Stimmprobe. Die Oberfläche sagt das an der Engine dazu und meldet es, bevor etwas rechnet.

Jede Engine bringt ihre eigenen Parameter mit. Die Oberfläche baut die Regler aus dieser
Deklaration (`ParamSpec` in `src/ttslab/engines/base.py`), eine neue Engine braucht deshalb
keine Änderung an der UI.

## Bedienung

- **Generieren**: Text eingeben, Stimmprobe und Sprache wählen, eine oder mehrere Engines
  ankreuzen. Jede Engine erzeugt ein eigenes Ergebnis. Über „Takes“ entstehen pro Engine mehrere
  Versuche mit verschiedenen Seeds. Ein fester Seed macht Läufe wiederholbar.
- **Stimmproben**: Aufnehmen oder hochladen, Transkript wird lokal per Whisper erstellt und ist
  editierbar. Das Transkript verbessert bei manchen Engines die Ähnlichkeit deutlich.
- **Vergleich**: Alle Ergebnisse mit Engine, Seed und Parametern. Mehrere auswählen, untereinander
  anhören, als Favorit markieren, als ZIP exportieren.

Stimmproben und Ergebnisse liegen unter `data/` und überleben einen Neustart.
Ein anderes Verzeichnis geht über `TTSLAB_DATA=/pfad uv run python app.py`.

## Tests

```bash
uv run pytest           # schnell, ohne Modelle
uv run pytest -m slow   # erzeugt echtes Audio mit allen drei Engines
```
