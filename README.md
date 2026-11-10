# ACI606 · Cloud Fleet Challenge — Fleet Office

Servicio Cloud para una flota virtual de robots (Ejercicio 4, ACI606 Cloud Computing).
Recibe telemetría, la **persiste en PostgreSQL (Neon)**, predice la condición de cada robot con un
**modelo de ML**, expone **métricas operacionales**, ejecuta **respaldos programados** y muestra
la flota en una **oficina isométrica estilo pixel art** (`/dashboard`).

| | |
|---|---|
| **URL pública** | **https://aci606-cloud-fleet-challenge.vercel.app** |
| **Panel** | `https://aci606-cloud-fleet-challenge.vercel.app/dashboard` |
| **Documentación interactiva** | `https://aci606-cloud-fleet-challenge.vercel.app/docs` |
| **Plataforma** | Vercel (PaaS, FastAPI serverless) + Neon (PostgreSQL gestionado). Alternativa: Render (`render.yaml`) |
| **Datos** | Tabla `telemetry` en Neon (región AWS US East); `backup_log` registra cada respaldo |

## Arquitectura

```text
   Simulador de flota / evaluador / navegador
                    |  HTTPS
                    v
   +-----------------------------------------+        +----------------------------+
   |  Vercel · FastAPI (app/main.py)          |  TLS   |  Neon · PostgreSQL          |
   |  /telemetry  /predict  /robots  /metrics | -----> |  telemetry (historial)      |
   |  /fleet  /coverage  /backups  /dashboard |        |  backup_log                 |
   |  models/model.joblib (cargado al inicio) |        |  PITR = respaldo físico     |
   +-----------------------------------------+        +----------------------------+
                    ^                                            ^
                    | despliegue automático en cada push a main  | respaldo lógico c/6 h
               GitHub (este repo)  ----------------------------  GitHub Actions
```

| Módulo | Responsabilidad |
|---|---|
| `app/main.py` | Rutas y contrato de la API, validación (datos inválidos → 422) |
| `app/db.py` | Persistencia con SQLAlchemy: PostgreSQL vía `DATABASE_URL` o SQLite local |
| `app/model.py` | Carga única de `models/model.joblib` e inferencia |
| `app/metrics.py` | Middleware ASGI: solicitudes, errores, latencia p50/p95, uptime |
| `app/backups.py` | Respaldo lógico, físico y copia offsite, con frecuencia configurable |
| `app/coverage.py` | Cobertura del área y puntos óptimos por robot |
| `ml/` | Dataset multi-flota y pipeline de entrenamiento con su informe |
| `dashboard/index.html` | Oficina de operaciones (canvas isométrico, sin dependencias) |

## Los 7 requisitos

| # | Requisito | Dónde se cumple |
|---|---|---|
| 1 | Ingesta persistente | `app/db.py`: historial completo sin límite; sobrevive reinicios y desconexiones |
| 2 | Entrenamiento de ML | `ml/train_model.py` → [`ml/reports/training_report.md`](ml/reports/training_report.md) |
| 3 | Servicio de inferencia | `POST /predict` con el contrato exacto |
| 4 | Panel de operación | `/dashboard`: robots, telemetría, predicción, batería, histórico, estado de la API |
| 5 | Observabilidad | `GET /metrics` |
| 6 | Trazabilidad Git | Ramas `feature/*` con merge `--no-ff` a `main` (`git log --graph`) |
| 7 | Ejecución remota | Vercel + Neon (ver [Despliegue](#despliegue-vercel--neon)) |

## Ejecutar en local

Requiere Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Sin `DATABASE_URL` la API usa SQLite en `data/fleet.db`. Para usar Neon en local, copia
`.env.example` a `.env`, o exporta la variable antes de iniciar.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

En otra terminal, la flota virtual:

```bash
python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 8
```

Abre <http://127.0.0.1:8000/dashboard> y <http://127.0.0.1:8000/docs>.

### Pruebas

```bash
python tests/smoke_test.py --final                       # contrato, historial, dashboard y 4xx
python tests/load_test.py --requests 1000 --concurrency 20  # p50/p95 de /predict desde el cliente
```

Resultado local (Windows, 1 proceso): 1000 predicciones concurrentes, 0 errores, p50 ≈ 34 ms, p95 ≈ 46 ms.

### Reentrenar el modelo

```bash
python ml/build_dataset.py      # 16 flotas x 3000 muestras, con fleet_id (usa el generador oficial)
python ml/train_model.py        # EDA, comparación, matriz de confusión, models/model.joblib
```

## API

| Ruta | Descripción |
|---|---|
| `GET /health` | Estado del servicio, de la base de datos y commit desplegado |
| `POST /telemetry` | Ingesta de una muestra; guarda también la predicción |
| `GET /robots` | Última muestra de cada robot |
| `GET /robots/{robot_id}/history` | Historial completo (`?limit=N` opcional, `?details=true` agrega la predicción) |
| `POST /predict` | Condición predicha (contrato abajo) |
| `GET /metrics` | Métricas operacionales |
| `GET /dashboard` | Oficina de operaciones |
| `GET /fleet`, `GET /fleet/summary` | Datos agregados para el panel |
| `GET /coverage` | Cobertura del área y punto óptimo de cada robot |
| `GET /backups`, `POST /backups/run?kind=logical\|physical` | Política, historial y ejecución de respaldos (`X-Backup-Token`) |

Contrato de `POST /predict`, sin cambios respecto al enunciado:

```json
{"robot_id": "R03", "timestamp": 1780000000.0, "x": 4.21, "y": 7.82, "speed": 0.63, "battery": 71.0,
 "motor_temp": 48.2, "vibration": 0.43, "motor_current": 3.7, "obstacle_distance": 0.42}
```

```json
{"robot_id": "R03", "predicted_status": "NORMAL", "confidence": 0.9999, "model_version": "v1", "inference_ms": 0.5}
```

Si falta un campo, hay un valor fuera de rango o llega un `NaN`/`Infinity`, la API responde **422** y nunca 5xx.
El manejador de errores no devuelve el valor inválido, porque un `NaN` no se puede serializar a JSON y produciría un 500.

## Modelo de ML

- **Datos:** `ml/build_dataset.py` ejecuta el generador oficial con 16 semillas, una flota por semilla, y agrega `fleet_id`.
- **Separación:** `GroupShuffleSplit` por `fleet_id`. La prueba usa 4 flotas que el modelo nunca vio, igual que la flota de evaluación.
- **Comparación** (GroupKFold 5 + prueba):

| Modelo | F1 macro CV | F1 macro prueba | Latencia por muestra |
|---|---|---|---|
| Línea base (clase mayoritaria) | 0.284 | 0.288 | — |
| **Regresión logística (4 features)** | **0.920** | **0.907** | **0.15 ms** |
| Random Forest (4 features) | 0.920 | 0.910 | 39 ms |
| HistGradientBoosting (4 features) | 0.920 | 0.903 | 2.2 ms |
| Regresión logística (6 features base) | 0.921 | 0.908 | 0.15 ms |

- **Elección:** regresión logística con `speed`, `motor_temp`, `vibration` y `motor_current`.
  - Iguala el F1 de los ensambles y es unas 250 veces más rápida que el Random Forest. Importa porque la evaluación mide p50/p95 y carga concurrente.
  - Se excluye `obstacle_distance` porque su información mutua con la clase es 0.
  - Se excluye `battery` porque funciona como un reloj: baja con el tiempo, igual que la salud del robot, pero no causa la falla. Una flota nueva llega con batería alta y puede traer robots CRITICAL desde el inicio.
  - `robot_id`, `timestamp`, `x` e `y` no describen la condición mecánica del robot.
- **Evidencia completa** (distribución de clases, medias por clase, información mutua, matriz de confusión y reporte por clase): [`ml/reports/training_report.md`](ml/reports/training_report.md).
- **Versionado:** `models/model.joblib` y `models/model_meta.json` están versionados a propósito, porque el PaaS no entrena. scikit-learn está fijado en `requirements.txt` para que el pickle sea compatible.

## Observabilidad (`GET /metrics`)

| Campo | Contenido |
|---|---|
| `telemetry_requests`, `prediction_requests`, `requests_total`, `requests_by_route` | Conteo de solicitudes |
| `errors_total`, `errors_4xx`, `errors_5xx`, `telemetry_errors`, `prediction_errors` | Errores |
| `avg_prediction_latency_ms`, `p50_prediction_latency_ms`, `p95_prediction_latency_ms` | Latencia de `/predict` medida en el servidor |
| `avg_inference_ms` | Tiempo puro del modelo |
| `uptime_seconds`, `instance_id` | Tiempo en funcionamiento; un reinicio cambia el `instance_id` |
| `telemetry_per_minute`, `predictions_by_status`, `telemetry_rows`, `robots` | Operación de la flota |
| `model` | Versión, F1 y features del modelo cargado |

En Vercel varias instancias de la API responden en paralelo. Cada una cuenta sus solicitudes en memoria y guarda esos contadores en la tabla `metrics_snapshot` de Neon después de cada solicitud, una vez enviada la respuesta. Si llegan varias solicitudes mientras guarda, se agrupan en la siguiente escritura, así que no se pierde ninguna cuenta.
`/metrics` suma los contadores de todas las instancias (`"scope": "service"`): solicitudes, errores y latencia promedio corresponden al servicio completo.
`uptime_seconds`, `p50/p95` y `telemetry_per_minute` son de la instancia que respondió, y su detalle queda en el bloque `instance`.

## Respaldos

| Tipo | Qué es | Frecuencia | Dónde |
|---|---|---|---|
| **Lógico** | Exporta las filas de `telemetry` a CSV.gz. No depende del motor y sirve para migrar SQLite → Neon | Cada 30 min en local o Render (`BACKUP_LOGICAL_MINUTES`); cada 6 h en la nube con GitHub Actions | `backups/logical/`, artefacto del workflow |
| **Físico** | Copia binaria de la base completa (API de backup de SQLite, consistente mientras la base recibe escrituras) | Cada 6 h (`BACKUP_PHYSICAL_MINUTES`) | `backups/physical/` |
| **Copia offsite del físico** | Cada respaldo físico se copia a otro disco o carpeta (`BACKUP_OFFSITE_DIR`). Si falla el disco, no se pierden juntos la base y su respaldo | Junto con cada respaldo físico | `BACKUP_OFFSITE_DIR` |
| **Físico en Neon** | Lo gestiona el proveedor: WAL continuo, restauración a un punto en el tiempo (PITR) y ramas | Continuo | Neon |

Se conservan los últimos `BACKUP_KEEP` archivos (10 por defecto). `GET /backups` muestra la frecuencia, el último respaldo
y el próximo, y el panel los muestra en la tarjeta **Respaldos**.

```bash
python -m app.backups logical
python -m app.backups physical
python -m app.backups restore backups/logical/telemetry_YYYYmmdd_HHMMSS.csv.gz   # no duplica filas
```

## Cobertura y puntos óptimos

La arena del simulador mide 10 × 10 m y es un toro: si un robot sale por un borde, entra por el opuesto.
`GET /coverage` y la capa **Puntos óptimos** del panel calculan:

1. **Cobertura real:** porcentaje de celdas de 1 m visitadas en la ventana de tiempo (60 s en el panel).
2. **Puntos óptimos:** con *n* robots, los centroides de una teselación de Voronoi centroidal (CVT), aproximada con k-means sobre la arena.
   Cada robot queda a cargo de una zona de igual área. Así se minimiza la distancia de cualquier punto de la arena al robot más cercano, que es el radio de cobertura.
3. **Asignación óptima:** el algoritmo húngaro (`linear_sum_assignment`) asigna cada robot a un punto de modo que la distancia total recorrida sea mínima.
   Usa distancia toroidal y calcula una ETA según la velocidad de cada robot.

## Panel: Fleet Office

Es una oficina isométrica clara con paredes de vidrio y robots en pixel art.

- **Planta de operaciones:** es la arena de 10 × 10 m del simulador. Los robots caminan según su x/y real, con movimiento continuo entre muestras.
- **Salas de vidrio alrededor:** cada una representa una pieza de la arquitectura:
  - **Centro de misiones:** puestos de API, Modelo ML y Métricas, con una pantalla en vivo de telemetría/min, predicciones, p95 y errores.
  - **Sala de servidores:** base de datos y respaldos.
  - **Zona de carga.**
- **Botón «Iniciar flota»:** pone en marcha 8 robots desde el mismo navegador, sin abrir una terminal. Usa las mismas fórmulas del simulador oficial (`simulator/core.py`, que no se modificó) y envía telemetría real a `POST /telemetry` cada segundo, así que los datos pasan por la validación, el modelo y la base de datos como los de cualquier cliente. Se detiene sola a los 20 minutos y no arranca si otro simulador ya está enviando los robots R01–R08.
- **Bordes de la arena:** el simulador hace que los robots salgan por un borde y entren por el opuesto, como en Pac-Man (`x %= 10`). El panel lo dibuja como un cruce: el robot se desvanece en un borde y aparece entrando por el otro, sin saltos.
- **Robots:** cada uno tiene nombre, rol y función. Su placa muestra el estado predicho (Trabajando, Alerta, Crítico o Sin señal) y la batería, y habla con globos de diálogo.
- **Ficha del agente** (clic en un robot): pestañas Telemetría (valores, historial de batería, temperatura y estados), Perfil (responsabilidades, herramientas y punto óptimo) y Actividad.
- **Actividad en vivo:** narra los eventos (cambios de estado, conexiones, respaldos, caídas de la API).
- **Panel lateral:**
  - **Pauta del profesor:** los 7 requisitos y las dimensiones de evaluación del PDF, con evidencia en vivo. El botón **Verificar ahora** comprueba desde el navegador el contrato de `/predict` (20 predicciones con su p50/p95), que los datos inválidos respondan 4xx y que el servicio siga activo.
  - Tablero de control: agentes activos, telemetría por minuto, predicciones, latencia p95, errores y cobertura.
  - Estado de la flota, con la distribución actual y la tendencia de 15 minutos.
  - Respaldos y modelo.

### Cuatro vistas: Oficina, Restaurante, Laboratorio y ¿Cómo funciona?

Arriba de la escena, un selector cambia entre cuatro vistas. El panel lateral (pauta, tablero, flota, respaldos y modelo) es el mismo en todas. Las tres flotan sobre un fondo de galaxia con constelaciones (Orión, Cruz del Sur, Casiopea y Osa Mayor).

- **Oficina** (`/dashboard`): la vista descrita arriba.
- **Restaurante** (`/dashboard#restaurante`): la misma plataforma y los mismos datos reales, contados como un restaurante.

| En el restaurante | En el sistema |
|---|---|
| Cliente en su mesa | Cada robot, que hace su pedido (telemetría) cada segundo |
| Meseros | La API: toman el pedido, lo revisan y lo llevan a la cocina |
| Chef | El modelo de ML, que «cocina» la predicción |
| Plato | La respuesta: verde normal, amarillo alerta, rojo crítico |
| Despensa · Caja · Recetario | Base de datos Neon · métricas · respaldos |
| Pedido rechazado | Datos inválidos (422) |

  Los meseros caminan de verdad: toman el pedido en la mesa, lo llevan al pase, el chef lo cocina y vuelven con el plato del color del estado predicho. La caja muestra las métricas en vivo.
- **Laboratorio** (`/dashboard#lab`): la simulación local descrita a continuación.
- **¿Cómo funciona?** (`/dashboard#como-funciona`): vista 2D sencilla que explica el sistema por dentro.
  - **Video del recorrido** (4 min, con voz, música y subtítulos): muestra todo el panel paso a paso. Está en `public/media/fleet-office-recorrido-voz.mp4`; Vercel lo sirve desde su CDN en `/media/…` (en local lo sirve la API). La voz se generó sin conexión con [Piper](https://github.com/OHF-Voice/piper1-gpl) (voz `es_MX-claude-high`, licencia Apache 2.0) y la música es original, sintetizada con código.
  - **Mapa del viaje de un dato**: robot → internet → API → revisión → modelo → base de datos → panel, con un sobre que viaja con datos reales.
  - **Ficha de cada pieza**: explicación simple o técnica, por qué está hecha así y un valor en vivo.
  - **«¿Qué pasa si…?»**: 15 casos; 9 tienen un botón que hace la petición real (422, 200, carga, `/health`, `/metrics`, historial).

### Modo laboratorio

El botón **Modo laboratorio** (o el enlace directo `/dashboard#lab`) cambia la oficina por una **simulación local en el navegador**. No envía datos a la API, no se guarda en Neon y no afecta al modelo ni a las métricas. La flota real la sigue moviendo el simulador oficial, que no se modificó.

- **Grilla variable:** de 8 × 8 a 24 × 24 celdas, con 2 a 12 robots y velocidad 1×, 2× o 4×.
- **Herramientas:** dibujar o borrar obstáculos arrastrando el mouse (o el dedo), y agregar o quitar puertos de carga.
- **Escenarios aleatorios:** "Obstáculos al azar" y "Nuevo escenario" usan una semilla, para que cada escenario sea distinto y reproducible. Los obstáculos al azar nunca dejan zonas aisladas.
- **Rutas:** cada robot calcula el camino más corto con búsqueda en anchura (BFS) y lo recalcula si se le pone un obstáculo delante. Explora prefiriendo las celdas menos visitadas.
- **Batería:** cada celda recorrida gasta batería. Antes de elegir un destino, el robot verifica que le alcanza para la ida y para volver al puerto más cercano. Si no, va a cargar; si el puerto está ocupado, espera en cola. Al completar la carga vuelve a explorar.

Es un complemento visual. Los datos que se evalúan (telemetría, predicción, métricas y persistencia) son los de la flota real.

## Despliegue: Vercel + Neon

1. **Neon:** crea un proyecto en <https://neon.tech> en la región AWS US East (N. Virginia), la misma región que la función de Vercel (`iad1`). Copia la cadena de conexión **pooled** (host `-pooler`).
2. **GitHub:** sube este repositorio. `.env`, `data/`, `backups/` y `.venv/` están en `.gitignore`.
3. **Vercel:** *Add New → Project → Import* del repositorio. Vercel detecta FastAPI en `app/main.py`. Python 3.14 se toma de `.python-version` y las dependencias de `requirements.txt`.
4. **Variables de entorno** en Vercel (*Settings → Environment Variables*):
   - `DATABASE_URL` = cadena de Neon. Acepta `postgresql://…?sslmode=require`; la app la convierte a `postgresql+psycopg://`.
   - `BACKUP_TOKEN` = valor secreto, para `POST /backups/run`.
   - `BACKUP_LOGICAL_MINUTES=360`, para que el panel muestre la frecuencia del workflow.
5. **Deploy.** Cada push a `main` vuelve a desplegar automáticamente.
6. **Respaldos programados:** en GitHub, agrega el secreto `DATABASE_URL` y la variable `ENABLE_BACKUPS=true` (*Settings → Secrets and variables → Actions*).

Alternativa en Render (la arquitectura de referencia del curso): *New → Blueprint* con `render.yaml`, y pega `DATABASE_URL` en el panel.

### Verificación del despliegue

```bash
python tests/smoke_test.py --api https://aci606-cloud-fleet-challenge.vercel.app --final                       # 1. contrato
python simulator/fleet_simulator.py --api https://aci606-cloud-fleet-challenge.vercel.app --robots 5 --steps 120  # 2. persistencia: >100 por robot
#    Redeploy en Vercel (o restart en Render) y comprobar que /robots/R01/history conserva todo
python tests/load_test.py --api https://aci606-cloud-fleet-challenge.vercel.app --requests 300 --concurrency 10   # 3. latencia y carga
```

Abre `/dashboard` desde otro dispositivo y revisa `/metrics` y los logs de Vercel.

### Resultado de la verificación (2026-11-02, desde Chile contra Vercel `iad1` + Neon)

| Verificación | Resultado |
|---|---|
| Contrato (`smoke_test.py --final`) | Todas las rutas 200, `/predict` con el contrato exacto, datos inválidos y `NaN` → 422, servicio activo después |
| Base de datos | `/health` → `"database": "POSTGRESQL"` (Neon) |
| Persistencia | 5 robots × 120 pasos: 119–120 registros por robot en Neon (2 de 600 envíos se perdieron por arranques en frío que superaron el *timeout* de 3 s del simulador) |
| Reinicio | Redeploy en Vercel (instancia nueva; `/health` → `commit` 88d7032): mismo número de registros y misma huella (hash) del historial de cada robot |
| Carga (`load_test.py`, 300 solicitudes, 10 concurrentes) | 0 errores. Cliente: p50 238 ms, p95 325 ms (incluye la red Chile–Washington). Servidor: p50 2,1 ms, p95 2,9 ms; modelo 1 ms |
Antes de la evaluación, llama a `GET /health` para despertar la base de Neon, que se suspende tras 5 minutos sin uso.

## Restricciones de la plataforma y cómo se resolvieron

| Restricción | Solución |
|---|---|
| Disco efímero o de solo lectura | Datos en PostgreSQL (`DATABASE_URL`). SQLite solo en local |
| Modelo fuera de Git | Se versiona `models/model.joblib` (excepción en `.gitignore`) |
| Secretos | Solo en variables de entorno. `.env.example` sin credenciales |
| Puerto asignado | Vercel lo gestiona. En Render: `--host 0.0.0.0 --port $PORT` |
| Arranque en frío | El modelo y las tablas se cargan en el `lifespan`. Se llama a `/health` antes de evaluar |
| Neon suspende conexiones | `pool_pre_ping=True`, `pool_recycle=300`. Sin *prepared statements* por el pooler |
| Sin hilos de fondo en serverless | Respaldos programados en GitHub Actions. El físico lo cubre el PITR de Neon |

## Estructura

```text
app/          API (main, db, model, metrics, backups, coverage)
dashboard/    Fleet Office (HTML + canvas)
ml/           build_dataset.py, train_model.py, reports/
models/       model.joblib + model_meta.json (versionados)
simulator/    simulador oficial (sin modificar)
tests/        smoke_test.py (--final), load_test.py
docs/         contrato, arquitectura y guía de defensa
```

## Trazabilidad Git

Cada requisito se desarrolló en su rama y se integró a `main` con `merge --no-ff`:
`feature/persistence`, `feature/ml`, `feature/inference`, `feature/observability`,
`feature/backups`, `feature/coverage`, `feature/dashboard` y `feature/deploy`.

```bash
git log --oneline --graph
```
