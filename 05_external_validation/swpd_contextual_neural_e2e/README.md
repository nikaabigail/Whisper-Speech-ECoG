# Контекстный neural end-to-end alternating — SWPD sub-01

Этот каталог — отдельный эксперимент. Он не изменяет зафиксированные результаты
`swpd_contextual_alternating_v2` и не подменяет последний контекстный L4-контроль.

## Что здесь действительно end-to-end

Нейросетевой декодер получает тот же контекст high-gamma, что и последний
эксперимент для Оссадчего: девять отсчётов `−200…+200 мс` по 127 каналам
(`9×127=1143`). Все параметры residual-ветви обучаются градиентами. Между
циклами при замороженной нейросети точным Procrustes-шагом обновляется
ортонормированный линейный проектор Whisper L4 `128→50`.

Whisper-энкодер заново не запускается: используются неизменяемые L4-признаки из
контекстного кэша. Поэтому «end-to-end» здесь означает совместную задачу от
ECoG-контекста до обучаемого 50-мерного Whisper-пространства, а не fine-tuning
самого Whisper.

## Три сравниваемых варианта

1. **Legacy cycle 0** — точное воспроизведение последнего линейного пути:
   `StandardScaler → PCA50 → OLS → L4 PCA50`. Его OLS сворачивается в skip-слой
   нейросети, а residual-выход инициализируется нулём.
2. **Fixed-Q neural** — та же нейросеть и тот же бюджет, но проектор Whisper
   остаётся исходным PCA50.
3. **Alternating-Q neural** — нейросеть обучается с фиксированным проектором,
   затем нейросеть замораживается и проектор обновляется точным ограниченным
   Procrustes-шагом.

Главное сравнение: `Alternating-Q neural − Fixed-Q neural`. Legacy нужен, чтобы
отделить эффект нейросети от эффекта обучаемого проектора.

## Протокол против утечек

- пять временных folds: test=`i`, validation=`i+1`, три остальных блока train;
- `StandardScaler`, PCA, MEL-probe и проектор обучаются только на train;
- внутри model-phase лучшая эпоха выбирается только по полному train MSE;
- validation читается один раз на конце цикла и выбирает цикл;
- первый neural-cycle общий, поскольку оба варианта ещё имеют один `Q0`;
- в последующих циклах fixed и alternating получают одинаковое число эпох;
- для каждого fold его собственная test-роль не участвует в transforms, loss,
  validation или selection; при cross-validation тот же физический блок может
  законно быть train/validation в другом fold;
- test открывает отдельная команда, проверяющая fingerprints и SHA-256;
- checkpoint при resume всегда загружается на CPU, поэтому сохранённый CPU RNG
  не переносится ошибочно на CUDA.

По умолчанию: seeds `1,2,3,4,42`, пять циклов по десять эпох. Мультисид здесь
нужен: в отличие от линейного OLS/SVD-контроля residual-сеть имеет случайную
инициализацию и стохастический порядок мини-батчей. Статистической единицей для
вывода о популяции всё равно остаётся пациент, а не seed или fold.

## Запуск fit-only в фоне

```powershell
Set-Location "<repo>\05_external_validation\swpd_contextual_neural_e2e"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_fit_background.ps1
```

Однократная проверка состояния:

```powershell
.\scripts\watch_fit.ps1
```

Непрерывный просмотр (Ctrl+C останавливает только просмотр):

```powershell
.\scripts\watch_fit.ps1 -Follow
```

Сначала рекомендуется отдельный диагностический каталог:

```powershell
.\scripts\start_fit_background.ps1 `
  -RunDir "C:\WhisperECoG_Work\SWPD\runs\contextual_neural_e2e_sub01_smoke" `
  -DiagnosticSmoke
```

Диагностический запуск никогда не считается production и не разрешает test.

## Единственное открытие test после фиксации

Только после строки `FIT COMPLETE` для production-каталога:

```powershell
.\scripts\run_evaluate_frozen.ps1
```

Итоговый отчёт содержит результаты по каждому fold и seed, затем сначала
усредняет folds внутри seed. Seed-вариативность описывает устойчивость
оптимизации на одном development-пациенте и не заменяет межпациентную статистику.

На RTX 5070 Laptop проверочный полный epoch одного `fold×seed` вместе с
подготовкой train-only PCA занял около 18 секунд. Производственный прогон содержит
25 комбинаций `fold×seed` и две compute-matched ветви; ориентир — примерно
45–90 минут, но фактическое время зависит от питания/охлаждения ноутбука.

## Публичный контракт `core.py`, используемый runner-ами

Runner ожидает от `core.py` следующие имена:

- `ContextualResidualDecoder(context_steps, channels, output_dim)`, методы
  `forward`, `initialize_legacy_skip`, `architecture_receipt`;
- `Standardizer`, `PCATransform`, `TargetSearchSpace`, `AffineMap`;
- `fit_affine`, `project_scores`, `exact_projector_update`;
- `fold_legacy_pipeline`, который сворачивает train-only neural
  StandardScaler/PCA50/OLS в точный affine skip;
- `common_mel_metrics`, `score_and_full_target_metrics`, `mse`;
- проверки `projector_receipt` и детерминированные RNG/save-load helpers.

Этот API намеренно узкий: runners отвечают за split/test-gate и provenance, а
`core.py` — за математику и нейросетевую архитектуру.
