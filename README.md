# Cognivision – Proyecto IS701
Clasificación de **estados cognitivos y mentales** a partir de rostros usando **DeepFace** (embeddings faciales) + un **modelo ML** propio (Random Forest / MLP) y un **frontend** para interacción en tiempo real.

> Monorepo con `apps/` (Django + Next.js) y `packages/` para módulos compartidos.

---

## 🧭 Tabla de contenidos
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Estructura del repo](#estructura-del-repo)
- [Configuración rápida](#configuración-rápida)
- [Backend · Django (apps/cognivision)](#backend--django-appscognivision)
- [Frontend · Next.js (apps/web)](#frontend--nextjs-appsweb)
- [Correr todo a la vez](#correr-todo-a-la-vez)
- [Variables de entorno](#variables-de-entorno)
- [Flujo de desarrollo](#flujo-de-desarrollo)
- [Roadmap](#roadmap)
- [Preguntas frecuentes](#preguntas-frecuentes)

---

## Arquitectura
- **Extracción de características**: `DeepFace` para obtener *embeddings* faciales.
- **Clasificador**: modelo supervisado (RandomForest / MLP) entrenado con etiquetas cognitivas:
  - estrés, fatiga mental, ansiedad visible, cansancio emocional, alta concentración, estado relajado/óptimo.
- **Backend (Django)**: API que orquesta DeepFace + modelo entrenado.
- **Frontend (Next.js)**: UI para capturar cámara/webcam, visualizar emoción + estado cognitivo y mostrar un consejo personalizado.
- **Packages**: utilidades compartidas (tipos, clientes, helpers).

---

## Requisitos
- **Python 3.10+**
- **Node.js 18+** y **pnpm 8+**
- **Git**
- (Opcional) **FFmpeg** y **OpenCV** si se procesa video localmente.

> Windows: ejecutar en **PowerShell**; para Python se recomienda `python -m venv .venv` y activar el entorno antes de usar Django.

---

## Estructura del repo
```
/apps
  /cognivision        # Backend Django
    manage.py
    /cognivision      # settings/urls/wsgi
    db.sqlite3
    /.venv            # entorno virtual de Python (local)
  /web                # Frontend Next.js
    next.config.js
    package.json
/packages             # Módulos compartidos (futuros)
pnpm-workspace.yaml
turbo.json
```

---

## Configuración rápida
```bash
# 1) Instalar dependencias del frontend (monorepo)
pnpm install

# 2) Backend (crear/activar venv + instalar Django)
cd apps/cognivision
python -m venv .venv
.\.venv\Scripts\Activate   # Windows (PowerShell)
# source .venv/bin/activate  # macOS/Linux
pip install django

# 3) Migraciones y levantar API
python manage.py migrate
python manage.py runserver 8000

# 4) En otra terminal, levantar el frontend
cd apps/web
pnpm dev
```
## 🛠 Configuración del entorno en VS Code (Windows/Linux/macOS)

Para que VS Code reconozca correctamente las dependencias instaladas (como `deepface`, `numpy`, etc.), sigue estos pasos después de crear tu entorno virtual:

### ✅ 1. Seleccionar el intérprete de Python correcto

1. Abre la paleta de comandos con `Ctrl + Shift + P`

2. Escribe y selecciona:  
Python: Select Interpreter

3. Si usas entorno virtual en `apps/api/.venv`, elige:
 # En Windows:
 <ruta_del_proyecto>/apps/api/.venv/Scripts/python.exe 
# En Linux/macOs:
<ruta_del_proyecto>/apps/api/.venv/bin/python

> Si no aparece, haz clic en `Enter interpreter path...` → luego `Find...` y navega hasta la ruta del entorno especificado anteriormente.

---

### ✅ 2. (Opcional) Agregar paths adicionales a Pylance

Para que los archivos en `packages/` también reconozcan los imports (como `deepface`), crea o edita el archivo `.vscode/settings.json`:

```json
{
"python.analysis.extraPaths": [
 "./apps/api",
 "./apps/api/.venv/Lib/site-packages"
]
}
```
---

## Backend · Django (`apps/api`)

### 1) Activar entorno virtual
**Windows (PowerShell)**
```powershell
cd apps/api

python -m venv .venv

.\.venv\Scripts\Activate
```

### 2) Dependencias mínimas
```powershell
pip install django
pip install tf-keras
```
> Próximamente (cuando integres el modelo): `pip install deepface opencv-python-headless numpy scikit-learn`

### 3) Migrar base de datos (SQLite por defecto)
```powershell
python manage.py migrate
```

### 4) Ejecutar servidor de desarrollo
```powershell
python manage.py runserver 0.0.0.0:8000
```
- API local: http://127.0.0.1:8000/

### 5) Crear un superusuario (admin de Django)
```powershell
python manage.py createsuperuser
```

> **Notas**  
> - Si usarás módulos compartidos de `/packages`, añade su ruta en `settings.py` (ejemplo):
>   ```python
>   import sys
>   from pathlib import Path
>   BASE_DIR = Path(__file__).resolve().parent.parent
>   sys.path.append(str(BASE_DIR.parent / "packages"))
>   ```
> - Para CORS cuando consuma el frontend, instala y configura `django-cors-headers`.

---

## Frontend · Vite React (`apps/web`)

### 1) Instalar dependencias
```bash
cd apps/web
pnpm install
```


### 2) Ejecutar en desarrollo
```bash
pnpm dev
```
- App local: http://localhost:3000/


## Correr todo a la vez

### Opción A: comandos manuales (dos terminales)
- Terminal 1 (backend):
  ```bash
  cd apps/cognivision && .\.venv\Scripts\Activate && python manage.py runserver 8000
  ```
- Terminal 2 (frontend):
  ```bash
  cd apps/web && pnpm dev
  ```

---

## Variables de entorno
### Backend (Django)
Crear `.env` (y cargarlo con `python-dotenv` o en settings):
```
DEBUG=True
SECRET_KEY=changeme
ALLOWED_HOSTS=127.0.0.1,localhost
CORS_ALLOWED_ORIGINS=http://localhost:3000
```


## Flujo de desarrollo
1. **Entrenar/actualizar modelo ML** (fuera de este repo inicialmente).
2. Exportar el modelo (ej. `model.joblib`).
3. En el **backend**, cargar el modelo y exponer endpoints (`/api/predict`) que:
   - reciban imagen (o frame de cámara),
   - obtengan embeddings con DeepFace,
   - pasen embeddings al clasificador,
   - devuelvan `{emotion, cognitive_state, advice}`.
4. El **frontend** consume el endpoint y muestra resultados en tiempo real.

---

## Roadmap
- [ ] Endpoint `/api/health` y `/api/predict` en Django.
- [ ] Integración de `deepface` y `opencv-python-headless`.
- [ ] Entrenamiento + persistencia de modelo (`sklearn`/`joblib`).
- [ ] UI de cámara en Next.js + overlay de resultados.
- [ ] Sistema de recomendaciones por estado.
- [ ] Autenticación básica para panel profesional (telemedicina/RRHH).

---

## Preguntas frecuentes
**¿No me funciona `django-admin` en Windows?**  
Activa el entorno: `.\.venv\Scripts\Activate` y usa `python -m django startproject <nombre>` si lo prefieres.

**¿Puedo usar otra DB?**  
Sí. Instala el driver (`psycopg2-binary` para PostgreSQL) y ajusta `DATABASES` en `settings.py`.

---