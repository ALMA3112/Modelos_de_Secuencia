# Continuación de corales de Bach con encoder-decoder (LSTM, GRU, GRU con atención)

Laboratorio de Procesamiento de Lenguaje Natural, 2026 S02. Universidad Sergio Arboleda.

**Integrante:** Carlos Alberto Cardona Pulido

| Recurso | Enlace |
|---|---|
| Repositorio | https://github.com/ALMA3112/Modelos_de_Secuencia |
| Reproductor MIDI en línea | https://alma3112.github.io/Modelos_de_Secuencia/ |
| Experimento MLflow | `nlp-bach-alma` (experimento 3) |
| Modelo registrado | `bach-chorales-alma`, versión 1, alias `champion`, run de origen E4 (`07b72a7f0ee54e3795bbec621f6c4d47`) |
| MLflow y API | URL vigente el día de la sustentación (ver nota) |



## 1. Entorno

- AWS Academy Learner Lab, EC2 `t3.large` (2 vCPU, 8 GB), Ubuntu 24.04, solo CPU.
- Python 3.12.3, PyTorch 2.14.1 (CPU), MLflow 3.17.2, FastAPI 0.162.0, pretty_midi 0.2.11. Versiones completas en `requirements.txt`.
- MLflow Tracking y Model Registry en la misma instancia (SQLite como backend, artefactos servidos por HTTP), arrancados con `systemd` (`deploy/mlflow.service`, `deploy/api.service`).

## 2. Instalación y ejecución

```bash
git clone https://github.com/ALMA3112/Modelos_de_Secuencia.git && cd Modelos_de_Secuencia
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo cp deploy/mlflow.service deploy/api.service /etc/systemd/system/   # rutas para el usuario ubuntu
sudo systemctl daemon-reload && sudo systemctl enable --now mlflow

python scripts/download_data.py          # descarga JSB Chorales
python scripts/make_windows.py           # genera las ventanas una sola vez
python -m pytest tests -q                # pruebas

export EXPERIMENT=nlp-bach-alma
./scripts/run_e1_e3.sh                   # E1, E2, E3
python scripts/run_experiment.py --name E4 --arch gru --lr 0.002
python scripts/select_model.py           # tabla y seleccion por perdida de validacion
python scripts/eval_test.py              # test, UNA sola vez
python scripts/generate_midi.py          # 2 MIDI
python scripts/attention_map.py          # mapa de atencion de E3
python scripts/register_model.py         # log_model + alias champion
sudo systemctl enable --now api          # API en el puerto 8000
```

No hay credenciales en el repositorio. Los datos crudos, las ventanas y los checkpoints están en `.gitignore`; los checkpoints viven en MLflow.

## 3. Datos y ventanas

Se usa JSB Chorales (repositorio de Géron) con las particiones suministradas. Los valores de las notas se conservan sin recodificar, con vocabulario fijo de 0 a 127 y columnas `note0..note3`. El silencio 0 es una categoría válida: no se usa como relleno ni se excluye de la pérdida.

| Partición | Corales | Largo (pasos) | Desplazamiento | Ventanas (48 pasos) |
|---|---|---|---|---|
| train | 229 | 100 a 516 | 8 | 5.681 |
| valid | 76 | 128 a 576 | 16 | 982 |
| test | 77 | 128 a 640 | 16 | 1.011 |

Cada ventana toma 32 pasos de contexto (X) y los 16 siguientes (Y), dentro de un solo coral, y se descartan los fragmentos finales incompletos. Las ventanas se generan una vez (`data/windows.npz`) y se reutilizan en todos los experimentos. Los valores máximos observados son 81, 74, 69 y 66 (train), todos dentro del vocabulario. La fracción de silencio es baja: 0,13 a 0,22 % en train, 0,76 a 0,85 % en valid y 0,36 a 0,40 % en test (`artifacts/data_report.json`).

## 4. Modelos

Los tres modelos comparten la representación: una tabla `Embedding(128, 16)` aplicada a cada una de las cuatro voces y concatenada en orden de voz (64 dimensiones por paso). Encoder y decoder son unidireccionales, de una capa y con dimensión oculta 128 (`bach/models.py`).

| ID | Arquitectura | Parámetros entrenables |
|---|---|---|
| E1 | Encoder-decoder LSTM | 266.752 |
| E2 | Encoder-decoder GRU | 217.088 |
| E3 | Encoder-decoder GRU con atención aditiva (Bahdanau) | 364.800 |

- E1 y E2 inicializan el decoder con el estado final del encoder; en LSTM se transfieren el estado oculto y el de memoria.
- E3 agrega atención aditiva sobre los 32 estados de salida del encoder (consulta: estado oculto previo del decoder) y mantiene el resto de decisiones de E2.
- El primer paso del decoder recibe el último paso de X; los demás reciben el paso real anterior de Y (teacher forcing). En generación libre cada entrada es la predicción anterior.
- La salida tiene forma `(batch, 16, 4, 128)`, es decir, logits sin normalizar.

**Entrenamiento.** Adam, lr 1e-3, lote 64, máximo 20 épocas, paciencia 4 sobre la pérdida de validación, semilla 42 reiniciada al comenzar cada run. Pérdida: entropía cruzada promedio sobre todas las notas con logits `(batch*16*4, 128)` y sin softmax previo ni `ignore_index`. Validación en modo evaluación, sin gradientes y con teacher forcing. Las exactitudes (por nota y por paso completo, con argmax) se acumulan sobre toda la partición y se calculan **recargando el checkpoint de la mejor época**.

## 5. Experimentos (validación)

| Run | Arq. | Hidden | LR | Mejor época / ejecutadas | Pérdida val | Exactitud nota | Exactitud paso | Parámetros | Tiempo (s) |
|---|---|---|---|---|---|---|---|---|---|
| E1 | LSTM | 128 | 0,001 | 20/20 | 0,7483 | 0,7901 | 0,6001 | 266.752 | 102,3 |
| E2 | GRU | 128 | 0,001 | 19/20 | 0,6339 | 0,8158 | 0,6382 | 217.088 | 126,7 |
| E3 | GRU + atención | 128 | 0,001 | 16/20 | 0,6626 | 0,8073 | 0,6234 | 364.800 | 414,9 |
| **E4** | GRU | 128 | **0,002** | **11/15** | **0,6275** | **0,8176** | **0,6404** | 217.088 | 68,5 |

Todos los runs se ejecutaron en CPU (`t3.large`). Run IDs: E1 `f3259cc3...`, E2 `3047f501...`, E3 `b9bcb751...`, E4 `07b72a7f...`. Cada run tiene en MLflow sus parámetros, curvas de pérdida por época, métricas del mejor checkpoint, configuración, curva (`loss_curve.png`), número de parámetros, equipo de cómputo y checkpoint (`artifacts/results/` replica los resúmenes). Se hizo además una prueba técnica de una época en un experimento aparte, que se eliminó.

**Selección.** Se eligió **E4** por menor pérdida de validación entre E1 y E4.

**Cambio de E4.** Se partió de E2, la arquitectura con menor pérdida entre E1 y E3, y se cambió un solo hiperparámetro: la tasa de aprendizaje de 1e-3 a 2e-3. Motivo: E2 seguía mejorando en la época 19 de 20, y se quería saber si una tasa mayor convergía antes con el mismo tamaño de modelo. Se descartó aumentar la dimensión oculta porque E3 ya mostró que más capacidad sobreajusta en esta tarea. No se usó test para decidir. E4 mejoró a E2 en 0,0064 de pérdida de validación y entrenó menos épocas (mejor época 11, parada en la 15).

**E1 frente a E2.** El GRU baja la pérdida de validación en 0,114 con un 18,6 % menos de parámetros. Pero E1 alcanzó su mejor época en la 20 de 20, todavía mejorando, así que parte de su desventaja puede ser convergencia más lenta dentro del tope de 20 épocas y no solo el tipo de celda.

**E2 frente a E3.** La atención no mejoró la pérdida (0,6626 frente a 0,6339). E3 tiene un 68 % más de parámetros y mostró más sobreajuste: en su mejor época la brecha entre entrenamiento y validación fue de 0,166 (0,4967 y 0,6626), frente a 0,107 en E2 (0,5264 y 0,6339). Su entrenamiento fue unas 3,3 veces más lento porque el decoder con atención avanza paso a paso.

**Salvedad.** Cada configuración se entrenó con una sola semilla (42). Eso permite comparar estas ejecuciones, pero no demuestra que una arquitectura sea superior en general, y las diferencias pequeñas (como E4 frente a E2) pueden estar dentro de la variación entre semillas. Además, los modelos tienen tamaños distintos.

## 6. Resultados en test

Evaluado una sola vez, solo con el modelo elegido (E4, mejor checkpoint, época 11), después de la selección. Los resultados se registraron en el run E4 y en `artifacts/test_result.json`.

| Partición | Pérdida | Exactitud por nota | Exactitud por paso completo |
|---|---|---|---|
| Validación | 0,6275 | 0,8176 | 0,6404 |
| Test | 0,6545 | 0,8103 | 0,6194 |

Test es algo peor que validación, lo esperable porque el modelo se eligió con validación.

## 7. Generación y análisis musical

Con el mejor checkpoint de E4 se tomaron los primeros 32 pasos de dos corales distintos de validación (`chorale_229` y `chorale_230`, los dos primeros por nombre) y se generaron 32 pasos nuevos en cada uno, **sin teacher forcing**, en bloques de hasta 16 pasos. Al terminar un bloque se toman los últimos 32 pasos disponibles como nuevo contexto, se vuelve a codificar y se reinicia el decoder. Para cada voz se muestrea con `softmax(logits / T)`.

| Parámetro | Valor |
|---|---|
| Temperatura | T = 1,0 |
| Semilla aleatoria de generación | 123 |
| Tempo | 120 negras por minuto (cada paso dura 60/(4·120) = 0,125 s) |
| Archivos | `artifacts/generation/coral1_chorale_229.mid`, `coral2_chorale_230.mid` |

Cada MIDI contiene el contexto (4 s) y la continuación (4 s), con cuatro voces (una pista por voz) y 0 como silencio. Por convención de reproducción se fusionan las notas consecutivas iguales dentro de cada voz, porque la matriz no distingue una nota sostenida de una repetida. Se pueden oír en https://alma3112.github.io/Modelos_de_Secuencia/.

Estadísticas de la continuación de 32 pasos (`artifacts/generation/info.json`):

| | 229 generado | 229 real | 230 generado | 230 real |
|---|---|---|---|---|
| Silencios | 0 | 0 | 0 | 0 |
| Notas distintas por voz | 7, 6, 4, 9 | 3, 3, 6, 6 | 5, 5, 4, 8 | 4, 5, 7, 6 |
| Pasos iguales al anterior | 0,452 | 0,742 | 0,516 | 0,516 |
| Notas iguales a la anterior | 0,758 | 0,871 | 0,774 | 0,758 |
| Notas MIDI tras la fusión | 34 | 20 | 32 | 34 |

- **Repeticiones.** En el coral 229 la continuación generada repite menos que la real (0,452 frente a 0,742 de pasos iguales al anterior); en el 230 coincide con la real (0,516).
- **Variedad.** En el coral 229 usa más notas distintas por voz que el original (hasta 9 frente a 6), señal de más movimiento. En el 230 las cifras son parecidas. Con muestreo a T = 1,0 es esperable algo más de variación.
- **Silencios.** Ninguna continuación contiene silencios, coherente con su rareza en los datos (menos del 1 %). El modelo casi nunca los predice, así que el análisis de silencios es limitado.
- **Impresión musical.** Al escuchar los dos MIDI, las continuaciones suenan coherentes con el contexto. COMPLETAR (opcional: voces, saltos, estilo percibido).
- **Desempeño predictivo frente a calidad musical.** La exactitud mide cuánto coincide la predicción con una continuación concreta y con teacher forcing. En generación libre los errores se acumulan y se muestrea, por lo que la continuación no tiene por qué coincidir con la real y aun así sonar bien. A la inversa, una exactitud alta no garantiza fraseo o armonía agradables. La métrica no captura la calidad percibida.

## 8. Mapa de atención de E3

Con el mejor checkpoint de E3 (época 16), aunque no es el modelo elegido, se generó libremente sobre el coral `chorale_229` (T = 1,0, semilla 123) y se tomó el primer bloque de 16 pasos generados. El mapa tiene 16 pasos generados (eje Y) por 32 pasos de contexto (eje X, donde 1 es el más antiguo y 32 el último) y barra de color con el peso de atención: `artifacts/generation/attention_E3.png` (datos en `attention_E3.json`).

- En los 16 pasos generados, la posición con mayor peso es la 32, el último paso del contexto.
- Los últimos 4 pasos del contexto reúnen en promedio el 63,4 % del peso; los primeros 16 reúnen el 31,8 %.
- Las cinco posiciones con mayor peso medio son 32, 31, 30, 12 y 29.

Esto describe dónde reparte el peso el modelo; no se interpreta como una explicación causal de sus predicciones. El mapa corresponde a un solo fragmento y a un muestreo concreto.

## 9. Modelo registrado y API

**Registro.** El checkpoint de E4 se restauró en el modelo y se guardó con `mlflow.pytorch.log_model` (`serialization_format="pickle"`, `code_paths=bach/`), vinculado al run E4 y registrado como `bach-chorales-alma`, versión 1, con alias `champion` (`artifacts/registered_model.json`). La carga se verificó desde otro proceso y otro directorio (`scripts/verify_registry.py`, ejecutado desde `/tmp`) por la URI `models:/bach-chorales-alma/1`, sin pesos locales. Solo se cargan modelos del propio equipo.

**API (FastAPI, puerto 8000).** Al iniciar resuelve `champion` a una versión concreta y la carga una sola vez; no consulta MLflow por solicitud. Si cambia el alias hay que reiniciar el servicio (`sudo systemctl restart api`). Tiene exactamente dos endpoints; la conversión a MIDI se hace fuera de la API.

| Endpoint | Función |
|---|---|
| `POST /generate` | Recibe `seed` (32 filas x 4 enteros entre 0 y 127), `steps` (1 a 64) y `temperature` (0,5 a 1,5; por defecto 1,0). Devuelve `generated` (steps x 4, sin el contexto), `model` y `version` (cadena). |
| `GET /health` | Devuelve `{"status":"ok","model":...,"version":...}` cuando el modelo está cargado. |

Las entradas inválidas se rechazan con HTTP 422 antes de llamar al modelo. Ejemplo de solicitud desde un coral cargado en pandas:

```python
payload = {
    "seed": coro[["note0", "note1", "note2", "note3"]].iloc[:32].values.tolist(),
    "steps": 32,
    "temperature": 1.0,
}
requests.post("http://<IP>:8000/generate", json=payload)
```

Pruebas contra el servicio en ejecución:

| Caso | Resultado |
|---|---|
| Solicitud válida, `steps=32` | HTTP 200, 32 filas, versión "1" |
| `steps=33` (no múltiplo de 16) | HTTP 200, 33 filas |
| `seed` con 31 filas | HTTP 422 |
| Nota 128 | HTTP 422 |
| `GET /health` | `{'status': 'ok', 'model': 'bach-chorales-alma', 'version': '1'}` |

Las pruebas automáticas (`tests/test_api.py`, 14 casos) cubren además 33 filas de `seed`, filas de 3 notas, nota -1, `steps` 0 y 65, temperaturas 0,4 y 1,6, y `seed` ausente (todos 422), longitud y rango de la respuesta. La documentación interactiva está en `/docs`.

## 10. Estructura del repositorio

```
bach/        data.py, models.py, metrics.py, train.py, midi.py
api/         main.py (FastAPI)
scripts/     descarga, ventanas, experimentos, seleccion, test, generacion, registro
tests/       datos, modelos, metricas, MIDI, API
deploy/      mlflow.service, api.service
artifacts/   resultados, tabla, test, MIDI, mapa de atencion
docs/        reproductor MIDI (GitHub Pages)
```

## 11. Uso de inteligencia artificial generativa

Se utilizó **Claude (Anthropic)** para planificar el laboratorio por fases, escribir y depurar el código (datos, modelos, entrenamiento, generación, API y despliegue), redactar las pruebas automáticas y documentar. El código se ejecutó y verificó en la instancia, y los resultados reportados provienen de esas ejecuciones. No se usaron modelos preentrenados ni servicios externos de generación musical: los modelos se entrenaron con los datos del laboratorio.

## 12. Limitaciones

- Una sola semilla por configuración: las diferencias pequeñas entre ejecuciones no son concluyentes.
- E4 se decidió después de conocer E1 a E3, porque la guía exige partir de la arquitectura de menor pérdida de validación. Test no se usó para ninguna decisión.
- Todas las voces se reproducen con el mismo instrumento (piano), y se distinguen por altura.
- La IP pública de la EC2 cambia en cada sesión del lab.
