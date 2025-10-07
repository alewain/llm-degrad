# PLAN (Borrador)

## Diagnóstico (SOURCE_DIR=Archivos)
- Token HF hardcodeado y referencias a Colab/rutas (`/content`, `MyDrive`).
- Un único script de generación (`experimento.py`) con múltiples responsabilidades.
- Notebooks de análisis duplicados y pesados; no forman parte del pipeline de ejecución.
- PDF/Excel y artefactos pesados mezclados en la raíz.

## Estructura propuesta (TARGET_DIR=Repo_nuevo) y justificación
- `src/`: todo el código Python (módulos + entry point)
  - `src/model_io.py`: carga del modelo/tokenizer y restauración usando estrategia `subset_in_memory` (guarda en memoria solo el subset de parámetros a degradar).
  - `src/params_groups.py`: definición de grupos de parámetros (attn, mlp, embed) **específica para Gemma-3-4b** (34 capas hardcodeadas). Si se usa otro modelo, debe adaptarse manualmente.
  - `src/degradation.py`: métodos de degradación (mult_gauss, ablation, uni_quant) y utilidades asociadas. 
    - **Métodos eliminados:** `uni_quant_lineal` y `lognorm` (configs cfg10-cfg15) no se incluirán en la migración - no hubo experimentos finales publicados con estos métodos.
  - `src/generation.py`: funciones actuales de generación/IT wrapper (sin cambiar firmas/salida); `generate_text` unificada con flags opcionales para imagen.
  - `src/perplexity.py`: cálculo de perplejidad (opcional, desactivado por default).
  - `src/vram_utils.py`: medición VRAM y ajuste de batch; modo `dry-run` para estimación previa.
  - `src/image_utils.py`: carga de imagen y preparación de prompts para modelos multimodales (Cookie Theft).
  - `src/persistence.py`: guardado/retome en un único JSON por corrida, resistente a interrupciones.
  - `src/orchestrator.py`: `run_experiment` factoriza restaurar/perturbar/generar/persistir (misma lógica).
  - `src/utils.py`: utilidades comunes (`setup_logging`, `set_all_seeds`).
  - `src/run_experiment.py`: **entry point CLI** para ejecutar experimentos.
- `configs/experiment_configs.py`: definiciones de configuraciones usando dataclasses (perfiles de experimentos).
  - Usa `@dataclass` con type hints para validación automática
  - Permite composición (herencia de configs base)
  - Permite computaciones (e.g., `batch_size = min(n_prompts, max_batch_size)`)
  - Mantiene la expresividad de Python (operador `|` para merge, validaciones custom)
- `configs/prompts.py`: listas de prompts para los 3 experimentos principales (solo IT).
  - Mantiene las listas Python originales (más simple que JSON)
  - Tres listas: `dream_prompts_it` (~38), `iq_prompts_it` (~65), `cookie_theft_prompts_it` (~20)
  - Las configs de experimentos importan directamente desde este módulo
  - No incluye variantes pretrained (PT)
- `results/`: salidas JSON por corrida.
  - Nombres: `outputs_{degradation_method}_{modelo_seleccionado}_{nombre_extra_json}.json`
  - Ejemplo: `outputs_uni_quant_gemma-3-4b-it_2025_05_20_dreams.json`
  - `results/samples/`: muestras pequeñas (100-200 registros) versionadas para notebooks y documentación.
- `notebooks/`: análisis (importan desde `src/` o leen muestras de `results/samples/`).
- `docs/`: documentación y enlace a la tesis.

Racional: modularidad mínima para claridad/reutilización; ejecución 100% local; configuración por dataclasses Python para type safety y expresividad; perplejidad opcional y aislada (desactivada por default); imagen opcional (con ajustes automáticos de `max_seq_length`).

## Sistema de configuración (dataclasses Python)
**Justificación:** Se usa dataclasses de Python en lugar de archivos YAML por:
- **Type safety:** Validación automática de tipos con type hints
- **IDE support:** Autocompletado, detección de errores, refactoring
- **Expresividad:** Permite computaciones (`batch_size = min(n_prompts, max_batch_size)`), imports, condicionales
- **Composición clara:** Herencia de configs base, merge con operador `|` o `dataclasses.replace()`
- **Sin parser externo:** No requiere librerías adicionales ni validación manual
- **Pythonic:** Usa las herramientas estándar del lenguaje

**Estructura:**
- Clase base `ExperimentConfig` con todos los campos comunes y defaults
- Instancias específicas (e.g., `dreams_it`, `cookie_theft_it`) que heredan y overridean
- Función `get_config(name: str)` para obtener configs por nombre
- Permite override programático: `config.n_rep = 5` o `dataclasses.replace(config, n_rep=5)`

**Nota sobre batching:**
- **`max_batch_size`** (config): Límite superior para el tamaño del batch (default: 40). Define cuántos prompts pueden procesarse simultáneamente por restricciones de VRAM.
- **`n_prompts`** (runtime): Cantidad total de prompts en la lista de la config. Siempre se procesan TODOS los prompts disponibles.
- **`batch_size`** (runtime): Tamaño efectivo del batch, calculado como `min(n_prompts, max_batch_size)`.
- Si `n_prompts <= max_batch_size`: se procesan todos en un solo batch.
- Si `n_prompts > max_batch_size`: se dividen en múltiples batches de tamaño `max_batch_size`.
- Ejemplo: Con 100 prompts y `max_batch_size=40` → 3 batches: [40, 40, 20]
- **Eliminado:** El parámetro `cant_batches` del código original (era confuso y redundante).

## Plan de migración (fases)
1) Fase 1 – Esqueleto y configuración
- Crear estructura `src/`, `configs/`, `results/`, `logs/`, `docs/`.
- Implementar utilidades en `src/utils.py`:
  - `setup_logging()`: logging estándar dual (consola + archivo)
  - `set_all_seeds(seed)`: setea todas las seeds de RNG (random, numpy, torch, torch.cuda) para reproducibilidad
- Migrar configuraciones de diccionarios Python a `configs/experiment_configs.py` usando dataclasses:
  - Crear clase base `ExperimentConfig` con todos los campos comunes
  - Crear configs específicas heredando de la base (e.g., `DreamsITConfig`, `CookieTheftITConfig`)
  - Usar `dataclasses.replace()` o el operador `|` para overrides
  - Incluir función helper `get_config(name: str) -> ExperimentConfig` para obtener configs por nombre
- Mover funciones de generación a `src/generation.py` (sin cambiar firmas/salida).
- Reemplazar `print()` por `logging.info()`, `logging.warning()`, `logging.error()` según corresponda.
- Reemplazar bloques de seed duplicados por llamadas a `set_all_seeds()`.
- Extraer listas de prompts a `configs/prompts.py`:
  - Migrar las listas desde `experimento.py` (líneas 5-197) manteniendo la estructura Python
  - **Experimentos a migrar (solo IT):**
    - `dream_prompts_it_nuevo` → `dream_prompts_it` (38 prompts de narración de sueños)
    - `IQ_prompts_IT` → `iq_prompts_it` (math + language + logic + factual + creativity, ~65 prompts)
    - `cookies_it_mas` → `cookie_theft_prompts_it` (20 prompts para descripción de imagen)
  - **No se migran:** `open_prompts` (PT), `pt_cookie_prompts` (PT), ni ninguna variante pretrained
  - Las configs en `experiment_configs.py` importan directamente: `from configs.prompts import dream_prompts_it, iq_prompts_it, cookie_theft_prompts_it`
  - Eliminar función `load_prompts()` de `src/utils.py` (ya no es necesaria)
- Documentar en `README` el uso de `HF_TOKEN` por variable de entorno.

2) Fase 2 – IO de modelo y restauración
- Implementar `model_io.py` con estrategia `subset_in_memory`: guarda el subset de parámetros degradables en memoria CPU al inicio del experimento y restaura desde ahí antes de cada nivel de degradación.
- Añadir docstrings/typing y logs mínimos.

3) Fase 3 – Degradación, VRAM y persistencia
- Mover degradaciones a `degradation.py` y limitar métodos a: `mult_gauss`, `ablation`, `uni_quant`.
- **Eliminar completamente:** 
  - Métodos `uni_quant_lineal` y `lognorm` (no se portean)
  - Configs cfg10-cfg15 (no se migran a dataclasses)
- **Perplejidad (opcional y aislada):**
  - Mover a módulo separado `src/perplexity.py` con función `evaluate_perplexity(model, tokenizer, text)`
  - Campo en config: `compute_perplexity: bool = False` (desactivado por default)
  - Campo en config: `perplexity_text: str = ""` (texto de evaluación, solo si compute_perplexity=True)
  - Si activado, se calcula una vez por nivel de degradación (no por repetición)
  - Resultado se guarda en campo opcional `perplexity` del JSON de salida
- `vram_utils.py`: documentar umbrales y agregar `dry-run`.
- `persistence.py`: JSON único por corrida, resistente a interrupciones.
  - Al inicio, carga el JSON existente (si existe) y construye set de prompts ya computados
  - Identifica prompts faltantes comparando con la combinación (param_group, std_dev, repeat_index, degradation_method, prompt_text)
  - Solo ejecuta los prompts que faltan
  - Guarda periódicamente (cada X prompts procesados) para minimizar pérdida en caso de interrupción
  - Permite retomar experimentos interrumpidos sin re-ejecutar trabajo ya hecho

4) Fase 4 – Orquestador y entry point
- Factorizar `run_experiment` en `orchestrator.py` y crear entry point CLI en `src/run_experiment.py`.
- Mantener comportamiento y nombres externos.

5) Fase 5 – Notebooks y muestras (FUTURO)
- **Nota:** La migración de notebooks existentes se abordará en una fase posterior, fuera del alcance inicial.
- Los notebooks actuales (34d, 37c, 39a, etc.) permanecen en `Archivos/` sin modificar por ahora.
- **Objetivo futuro:** Crear notebooks de ejemplo que importen desde `src/` y lean muestras de `results/samples/`.
- Proveer muestras mínimas de resultados (100-200 registros por método) en `results/samples/` para versionar.

## Mejoras futuras (fuera del alcance inicial)
- **Detección automática de capas:** Implementar función que detecte el número de capas del modelo automáticamente, eliminando el hardcoding en `params_groups.py` (actualmente específico para Gemma-3-4b con 34 capas).
- **Validación de arquitectura:** Agregar validación que verifique que el modelo cargado coincide con los grupos de parámetros definidos.
- **Soporte multi-modelo:** Extender soporte a otras familias de modelos (LLaMA, Mistral, etc.).
- **Flag `force_run`:** Implementar flag opcional para forzar re-ejecución de prompts ya computados (sobrescribir resultados existentes). Actualmente, el sistema siempre retoma desde donde quedó, sin opción de sobrescribir. El flag permitiría:
  - `force_run=False` (default): comportamiento actual, solo ejecuta prompts faltantes
  - `force_run=True`: re-ejecuta todo, ignorando resultados previos
  - Útil para debugging o cuando se quiere regenerar outputs con cambios menores en el modelo/configuración

## Contrato de salida (actualizado)

**Compatibilidad:** Todos los campos del JSON actual se mantienen para no romper análisis existentes. Los campos nuevos se agregan de forma incremental.

**Campos actuales (mantenidos):**
- `timestamp`, `model_name`, `prompt_group`, `prompt_id`, `prompt_text`, `output`, `std_dev`, `repeat_index`, `temperature`, `do_sample`, `param_group_name`, `seed`, `duration`, `tokens`, `degradation_method`, `usar_4bit`
- **Nota:** `prompt_group` se mantiene por compatibilidad y se derivará del nombre de la config o se especificará explícitamente
- **Eliminados:** `force_run` (ver "Mejoras futuras"), `usar_data_parallel` (DataParallel no se usa en esta versión)

**Campos nuevos (a agregar):**
- `experiment_id`: identificador único de la corrida (formato: `{degradation_method}|{model_name}|{name_suffix}`)
- `config_name`: nombre de la configuración (dataclass) usada
- `model_variant`: "it" (siempre - solo se soporta instruction-tuned en esta versión)
- `tokens_in`: tokens del prompt (nuevo)
- `tokens_out`: alias de `tokens` (para claridad, mismo valor)
- `level_value`: alias de `std_dev` (para consistencia semántica)
- `level_index`: índice del nivel de degradación en el rango (nuevo)
- `batch_size_effective`: batch_size usado en esa generación (nuevo)
- `device`: "cuda:0", "cuda:1", etc. (nuevo)
- `dtype`: "float16", "float32", "bfloat16" - tipo de dato de los tensores en memoria durante generación (nuevo)
- `load_4bit`: booleano - indica si el modelo fue cargado con cuantización de 4 bits (nuevo)
  - `false` (default): modelo cargado en precisión completa (float16/float32)
  - `true`: modelo cargado con cuantización int4 para reducir uso de VRAM
- `restore_strategy`: "subset_in_memory" (único valor implementado: restauración desde memoria)
- `gen_params`: diccionario con {temperature, do_sample, max_new_tokens, top_k, top_p} (nuevo)
- `image_used`: booleano (nuevo)
- `image_filename`: opcional (nuevo)
- `vram_usage_percent`: opcional (nuevo)
- `perplexity`: opcional - solo presente si `compute_perplexity=True` en config. Calculado una vez por nivel de degradación.

**Campos eliminados:**
- `entorno`: ya no distinguimos colab/local (todo es local)

## Imagen (Cookie Theft)
- Si `image.enabled=true`: `max_seq_length` se ajusta automáticamente a 1024 (vs. 512 por defecto).
- La función `generate_text` acepta parámetros opcionales `processor` e `image`; procesa condicionalmente según presencia de imagen.
- Manejo aislado en `image_utils.py` para carga y preparación, integración en `generate_text` mediante flags.

## Política de restauración

**Estrategia implementada:** `subset_in_memory`

- El modelo se restaura **al iniciar cada repetición** (antes de aplicar degradación), una vez por cada combinación de `(nivel_degradación, repeat_index)`.
- Al inicio del experimento, se guarda en memoria CPU (como `.clone()`) el subset de parámetros que serán degradados.
- Antes de cada repetición, se restaura el modelo desde este subset guardado en memoria (rápido, sin I/O de disco).
- Cada experimento es independiente y crea su propio baseline en memoria al cargar el modelo.
- Esta estrategia es suficiente para la mayoría de casos y permite reintentos sin necesidad de recargar el modelo completo.

## Utilidades y funciones auxiliares (src/utils.py)

### Logging estándar
- **Configuración dual:** consola + archivo en `logs/<experiment_id>_<timestamp>.log`
- **Formato simple:** `%(message)s` (sin timestamps adicionales, ya están en los mensajes)
- **Nivel por defecto:** INFO (configurable a DEBUG para desarrollo)
- **Encoding UTF-8** para soportar emojis (✅, ⚠️, ❌)
  - En Windows, se requiere `sys.stdout.reconfigure(encoding='utf-8')` al inicio del script
  - Esto previene errores de encoding con caracteres especiales en consola
- **Captura todo lo que aparece en pantalla:** progreso, tiempos, warnings, VRAM, errores
- **No duplica** los outputs generados (esos van solo al JSON)
- Ejemplo de implementación:
  ```python
  import logging
  import sys
  import os
  
  def setup_logging(log_filename):
      # Configurar encoding UTF-8 para consola (necesario en Windows)
      if sys.stdout.encoding != 'utf-8':
          sys.stdout.reconfigure(encoding='utf-8')
      
      os.makedirs("logs", exist_ok=True)
      formatter = logging.Formatter('%(message)s')
      
      file_handler = logging.FileHandler(log_filename, encoding='utf-8')
      file_handler.setFormatter(formatter)
      
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(formatter)
      
      logger = logging.getLogger()
      logger.setLevel(logging.INFO)
      logger.addHandler(file_handler)
      logger.addHandler(console_handler)
  
  # Uso: reemplazar print() por logging.info(), logging.warning(), logging.error()
  ```

### Seeds y reproducibilidad
- **Función centralizada:** `set_all_seeds(seed: int)` setea todos los RNG (Random Number Generators)
- **Generadores seteados:**
  - `random.seed()`: módulo random de Python stdlib
  - `np.random.seed()`: generador de NumPy
  - `torch.manual_seed()`: PyTorch CPU
  - `torch.cuda.manual_seed_all()`: PyTorch CUDA (todas las GPUs disponibles)
- **Uso:** Se llama al inicio del experimento y antes de cada repetición con `seed_base + repeat_index`
- Ejemplo de implementación:
  ```python
  import random
  import numpy as np
  import torch
  
  def set_all_seeds(seed: int):
      """Setea todas las seeds de RNG para reproducibilidad completa."""
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
  ```

## Manejo de VRAM
- **Manejo preventivo (mantener lógica actual):**
  - Si VRAM > 95%: guardar resultados y abortar experimento con mensaje de error.
  - Si VRAM > 90%: advertencia y pausa breve (1 segundo).
  - Si VRAM < 40%: incrementar `batch_size` automáticamente (hasta `max_batch_size`).
  - El `batch_size_effective` se registra en cada resultado.

## Riesgos y mitigaciones
- Configuraciones Python inválidas → type hints y validación automática con dataclasses, más ejemplos claros.
- Variabilidad de VRAM → `dry-run` + logging de `batch_size_effective`.
- Fugas de secretos → sólo `HF_TOKEN` por entorno, `.gitignore` para caches/artefactos.
- Reproducibilidad → semillas fijadas por corrida + registro en `gen_params`.
- Notebooks antiguos incompatibles → mantener campos JSON originales para compatibilidad backward.
