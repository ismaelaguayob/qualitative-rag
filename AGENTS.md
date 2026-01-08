# Repository Guidelines

## Project Structure & Module Organization
- `app.py` houses the Streamlit app and the full RAG-in-cascade pipeline.
- `chroma_db/` stores the persistent Chroma vector database.
- `data_temp/` is a working area for uploaded files, extracted text, and logs.
- `data_temp/debug/` centralizes runtime logs for troubleshooting across sessions.
- `data_temp/dev_sessions/` should be updated at the end of each dev session with key changes and next steps.
- `documentation/` contains integration notes for providers and packages in use.
- `plans/` contains feature plans and implementation drafts.
- `requirements.txt` lists Python dependencies.

## Build, Test, and Development Commands
- `python3 -m venv venv` creates the virtual environment.
- `source venv/bin/activate` activates the environment.
- `pip install -r requirements.txt` installs dependencies.
- `streamlit run app.py` starts the local app.

## Coding Style & Naming Conventions
- Use Python 3.11+.
- Indentation: 4 spaces; prefer descriptive snake_case for variables and functions.
- Constants are upper snake case (e.g., `CHROMA_PATH`).
- Keep user-facing strings in the `I18N` dictionary and reuse `get_text`.

## Testing Guidelines
- No automated test suite is configured yet. If you add tests, document the command(s) here.
- Prefer naming tests as `test_*.py` and place them in a `tests/` directory.

## Commit & Pull Request Guidelines
- No explicit commit convention found. Keep messages short and imperative (e.g., "Add Streamlit sidebar settings").
- PRs should include a clear description, setup notes, and screenshots of UI changes.

## Security & Configuration Notes
- Required env vars live in `.env`: `GROQ_KEY` and `VOYAGE_KEY`.
- Do not commit credentials or uploaded data artifacts from `data_temp/`.
  
## Debug & Session Logs (Important)
- Always inspect `data_temp/debug/` when troubleshooting; logs are the source of truth for runtime issues.
- After each dev session, document the main changes and next steps in `data_temp/dev_sessions/YYYY-MM-DD.md` to keep continuity across chats.
