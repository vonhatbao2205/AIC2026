# Superseded

The first packaging attempt, kept only so nothing was thrown away silently.
Everything here is replaced by `docker/`:

| Was | Now |
|---|---|
| `Dockerfile` (repo root) | `docker/Dockerfile` — plus Supabase build args |
| `.dockerignore` (repo root) | `docker/Dockerfile.dockerignore` |
| `docker-compose.yml` (repo root) | `docker/compose.dev.yml` |
| `scripts/package.sh` | `docker/build-bundle.sh` |
| `scripts/dist/` | `docker/bundle/` — plus the Windows auto-installer |

Two live build definitions is how they drift apart, so the root copies had to
go. Safe to delete this whole directory once you have shipped one bundle from
the new path.
