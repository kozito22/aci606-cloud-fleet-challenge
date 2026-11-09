# Guía para la defensa técnica

Preguntas probables y cómo responderlas con lo que hay en el repositorio.

## Arquitectura y Cloud

**¿Por qué PaaS y no IaaS?**
En PaaS nosotros administramos el código y los datos, y el proveedor opera el sistema operativo, el runtime y la red (NIST SP 800-145).
No mantenemos servidores; el despliegue sale de GitHub y la configuración vive en variables de entorno.

**¿Por qué Vercel y no Render, que es la referencia del curso?**
Las dos son PaaS. Vercel ejecuta FastAPI como función serverless: no se duerme 15 minutos como Render free, despliega en cada push y entrega HTTPS.
El costo es que no hay disco persistente ni hilos de fondo. Por eso los datos van a Neon y los respaldos programados a GitHub Actions.
Si hiciera falta, el repositorio también incluye `render.yaml`.

**¿Por qué Neon?**
Es PostgreSQL gestionado y gratuito, sin expiración (la base gratuita de Render expira a los 30 días).
Tiene conexión con TLS, respaldo físico continuo (PITR) y suspende la base cuando no hay uso; para reconectar usamos `pool_pre_ping=True`.

**¿Qué pasa si se reinicia el servicio?**
La telemetría está en Neon, no en la memoria del proceso. Tras el reinicio, `/robots/{id}/history` devuelve lo mismo.
Solo se reinician los contadores de `/metrics`; el cambio de `instance_id` lo deja en evidencia.

**¿Dónde están las credenciales?**
Solo en las variables de entorno de Vercel y en los *secrets* de GitHub. `.env` está en `.gitignore` y `.env.example` no tiene valores reales.

## Persistencia

**¿Por qué SQLAlchemy Core y no un ORM completo?**
Las operaciones son simples (insertar, última muestra por robot, historial) y Core produce el mismo SQL en SQLite y en PostgreSQL.
Así, el mismo código corre en local y en la nube.

**¿Cómo se obtiene la última muestra de cada robot?**
`SELECT * FROM telemetry WHERE id IN (SELECT MAX(id) … GROUP BY robot_id)`, con el índice `(robot_id, id)`.

**¿El historial tiene límite?**
No. El starter guardaba 100 registros por robot y ahora se guarda todo; `?limit=N` es opcional.

## Validación y robustez

**¿Qué pasa con un `NaN`?**
Pydantic lo rechaza (`allow_inf_nan=False`) y la API responde 422. Además hubo que reemplazar el manejador de errores por defecto de FastAPI: este repetía el valor inválido en la respuesta, y un `NaN` no se puede convertir a JSON, así que terminaba en un 500. Lo encontramos probando.

**¿Y si el modelo falla durante la ingesta?**
La telemetría se guarda igual, con la predicción en `NULL`. La ingesta nunca falla por culpa del modelo.

## Machine Learning

**¿Por qué separar entrenamiento y prueba por `fleet_id`?**
Las muestras de un mismo robot están correlacionadas en el tiempo. Una separación aleatoria filtra información y da un F1 optimista.
Al separar por flota, la prueba imita a la flota no vista del profesor.

**¿Por qué F1 macro?**
Promedia el F1 de cada clase con el mismo peso. En nuestro dataset CRITICAL es el 76 %: la línea base de clase mayoritaria tiene 0.76 de *accuracy*, pero solo 0.29 de F1 macro.

**¿Por qué regresión logística?**
Empata en F1 con Random Forest y HistGradientBoosting (diferencias menores a 0.005), y es unas 250 veces más rápida que el Random Forest: 0.15 ms frente a 39 ms.
La clase depende de una salud interna que el simulador convierte casi linealmente en temperatura, vibración y corriente, así que un modelo lineal basta y además da probabilidades calibradas para `confidence`.

**¿Por qué quitar `battery` y `obstacle_distance`?**
`obstacle_distance` tiene información mutua 0 con la clase. `battery` solo baja con el tiempo, igual que la salud, así que la correlación es temporal y no causal.
Una flota nueva llega con batería alta y robots que pueden estar CRITICAL desde el inicio. Con 4 features el F1 es el mismo y el modelo es más robusto.

**¿Dónde está la matriz de confusión?**
En `ml/reports/training_report.md`. El error más común es entre WARNING y sus vecinos, NORMAL y CRITICAL, lo que tiene sentido: son estados contiguos de la misma salud.
NORMAL y CRITICAL casi nunca se confunden entre sí.

## Observabilidad

**¿Cómo se mide la latencia?**
Un middleware ASGI cronometra cada solicitud. Para `/predict` se guardan las últimas 2.000 mediciones y de ahí salen p50 y p95.
`inference_ms` mide solo el modelo; la diferencia con la latencia total es el costo de HTTP y de la validación.

**¿Por qué un middleware ASGI puro?**
`BaseHTTPMiddleware` agrega sobrecosto por solicitud, y la latencia es parte de la evaluación.

## Respaldos

**¿Diferencia entre respaldo lógico y físico?**
- **Lógico:** exporta los datos (filas a CSV). Es portable entre motores y versiones, se puede leer y restaurar de forma selectiva, pero es más lento con bases grandes.
- **Físico:** copia los archivos o páginas de la base tal cual. Restaura rápido y exacto, pero solo sirve para el mismo motor y versión.

**¿Cada cuánto se hacen?**
El lógico cada 30 minutos en local o en Render, y cada 6 horas en la nube con GitHub Actions. El físico cada 6 horas en SQLite; en Neon es continuo (WAL y PITR).
Se conservan los últimos 10 archivos. `GET /backups` y el panel muestran el último y el próximo.

**¿Para qué la copia offsite?**
Un respaldo guardado en el mismo disco que la base se pierde junto con ella. Copiar el respaldo físico a otro lugar aplica la regla 3-2-1: 3 copias, 2 medios distintos, 1 fuera del sitio.

## Cobertura

**¿Qué son los puntos óptimos?**
Para *n* robots, los centroides de una teselación de Voronoi centroidal de la arena. Cada robot cubre una zona de igual área y la distancia máxima de cualquier punto a su robot es el radio de cobertura.
El algoritmo húngaro asigna cada robot a un punto minimizando la distancia total, medida en un toro porque el simulador envuelve los bordes.
El simulador no se modificó: la API calcula el plan y el panel lo muestra.

**¿Por qué los robots salen por un borde y entran por el otro?**
Así funciona el simulador: después de moverse, aplica `x %= 10` e `y %= 10`, de modo que la arena es un toro, como en Pac-Man. No es un error de la API.
El panel dibuja la ruta más corta sobre el toro (`|dx| <= 5`) y muestra el cruce con un desvanecimiento, en vez de un salto.
La misma distancia toroidal se usa para asignar los puntos óptimos.

**¿Qué es el modo laboratorio? ¿Cambia algo de lo evaluado?**
Es una simulación local en el navegador para experimentar con obstáculos, tamaño de grilla y carga de batería. No envía datos a la API ni a Neon, y no toca el modelo: la flota real sigue siendo la del simulador oficial.
No sirve para evitar *data leakage*. Ese riesgo ya se controla en el entrenamiento separando las flotas, y el modelo no usa la posición (`x`, `y`).

**¿Cómo deciden los robots del laboratorio cuándo cargar?**
Cada celda gasta batería. Antes de ir a un destino, el robot calcula con BFS la ida y la vuelta al puerto más cercano. Si no le alcanza con reserva, va a cargar; si el puerto está ocupado, espera en cola.

## Demostración en vivo (checklist)

1. `GET /health` para despertar el servicio.
2. Iniciar el simulador contra la URL pública y mostrar `/dashboard` desde el celular.
3. Hacer *Redeploy* en Vercel (o *Restart* en Render) y mostrar que `/robots/R01/history` conserva todo.
4. `python tests/smoke_test.py --api <URL> --final`.
5. Mostrar `/metrics` y la tarjeta de respaldos.
