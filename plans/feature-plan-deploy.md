# Plan de Deploy Definitivo

Fecha: 2026-01-07

## Objetivo
Definir el camino de despliegue productivo para el proyecto, incluyendo arquitectura, base de datos, autenticación y operación en servidor propio.

---

## Decisiones clave (a validar)
1) **Framework UI**
   - Opción A: Mantener Streamlit y extender con autenticación, permisos, cache, async.
   - Opción B: Migrar a otro framework (por ej. FastAPI + frontend separado).

2) **Base de datos**
   - Migrar a **PostgreSQL** para multiusuario, historial de chats y metadatos.
   - Definir esquema: usuarios, sesiones, documentos, chunks, tópicos, chats.

3) **Infraestructura**
   - Servidor propio con Docker.
   - Automatización con GitLab Actions (CI/CD).

4) **Embeddings / LLMs**
   - Revisar proveedores más confiables / económicos.
   - Plan fallback local si no hay proveedor viable.

---

## Fases sugeridas

### Fase 1: Hardening del MVP
- Logging centralizado.
- Configuración clara de env vars.
- Gestión de errores y timeouts.

### Fase 2: Persistencia avanzada
- PostgreSQL + migraciones.
- Indexación persistente y versionada.
- Historial de chats por usuario.

### Fase 3: Autenticación y permisos
- Login, roles, permisos.
- Protección de datos sensibles.

### Fase 4: Deploy y CI/CD
- Dockerfile y docker-compose.
- Pipeline GitLab Actions (lint + build + deploy).
- Monitoring básico (uptime + logs).

---

## Entregables
1) **Documento de arquitectura** (diagramas + decisiones).
2) **Schema DB** (SQL + migraciones).
3) **Infraestructura Docker** (compose + secrets).
4) **Pipeline CI/CD** funcional.

---

## Tests manuales (dev)
- Usuario se autentica y crea sesión persistente.
- Historial de chat se guarda y puede reabrirse.
- Reiniciar contenedor sin perder índices/chats.
- Desplegar en servidor y validar acceso externo.

---

## Notas
- Coordinar con admin para permisos de Docker y despliegue en IP pública.
- Definir estrategia de backups (DB + vectores).
