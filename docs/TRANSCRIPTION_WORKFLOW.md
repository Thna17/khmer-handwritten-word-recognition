# Local Khmer Line Transcription Workflow

This workflow is for human transcription only. It does not train a model, alter crop images, alter source pages, or change crop IDs.

## Start the application

From the repository root:

```bash
source .venv/bin/activate
python scripts/transcribe_lines.py
```

The application opens at <http://127.0.0.1:8765/>. If the browser does not open automatically, copy that address into a browser. Stop the server with `Ctrl+C`; every submitted entry has already been saved.

## Transcribe a line

1. Confirm the displayed `crop_id`, writer, and source page.
2. Read the enlarged line image and enter only the visible Khmer text in `label_raw`.
3. Choose a status:
   - `untranscribed`: not handled yet; label must be blank.
   - `transcribed`: entered once but not independently checked; label is required.
   - `unclear`: cannot be read confidently; label must be blank.
   - `verified`: checked and accepted; label is required.
   - `rejected`: not a usable handwritten line and excluded from training.
4. Optionally enter reviewer notes and a confidence from `0` to `1`.
5. Select **Save & next**. The metadata CSV is atomically replaced after every save.

Use **Previous** and **Next** to navigate without saving. Unsaved edits trigger a warning. Use the writer and status menus to filter the queue. The table on the right reports progress for every writer.

## Critical labeling rules

- Enter the transcription exactly as it appears. The app does not normalize, trim, or rewrite Unicode.
- Never type `[UNCLEAR]`, `[ILLEGIBLE]`, `[UNKNOWN]`, or similar markers into `label_raw`.
- If a line cannot be read confidently, leave `label_raw` blank, select `unclear`, explain why in reviewer notes, and save.
- Do not set a blank line to `verified`; the application rejects that save.
- `label_normalized` is intentionally untouched. Unicode normalization and human verification happen in the later normalization stage.
- Previously rejected records do not appear in the default usable queue. Select the `rejected` filter only when reviewing them intentionally.

## Safe resume and backup behavior

Edits are written to the appropriate `data/handwritten_external/<writer>/line_metadata.csv`. Each save is written to a temporary file in the same directory, flushed to disk, and atomically moved into place. This prevents a partially written CSV if the app is interrupted.

Before a large review session, retain the latest checksum snapshot under `data/metadata/dataset_snapshots/`. Never run crop extraction over a writer directory while the transcription application is open.

## Run workflow tests

The tests use only temporary directories and synthetic images; they do not write labels into the real dataset:

```bash
python -m pytest -q tests/test_transcribe_lines.py
```
