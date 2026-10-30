# Just Allah backend agent instructions

This repository is the FastAPI backend for the Islamic sources RAG app. The current scaffold is intentionally minimal and should remain simple until the project reaches the later plan parts.

## Actual structure

- `app/main.py` is the active FastAPI application entry point.
- `pyproject.toml` declares the project and uses `uv` for dependency management.
- `uv.lock` is the lockfile for the project.
- `.env` is the local environment file for runtime configuration and should stay in the backend directory.
- There are no routers, services, database models, or migrations yet; add them only when later plan parts require them.

## Tooling and commands

- Use `uv` to install and run Python tooling for this project.
- The project currently targets Python 3.13 per `pyproject.toml`.
- The FastAPI app is minimal and does not yet include custom dependency injection or database setup.

## Conventions

- Keep the backend in a straightforward FastAPI layout.
- Prefer small, explicit modules instead of overbuilding the app.
- Keep configuration in `.env`; do not hardcode secrets or database URLs.
- Follow the project plan in order; do not start later migration or ingestion work before the earlier parts are complete.
- No authentication or write endpoints are present yet.

## Working style

- Keep validation focused and minimal.
- Preserve the existing scaffold unless a later plan step explicitly requires a new file or module.
- When adding new code, match the project’s later architecture rather than inventing a new structure for the sake of convention.
