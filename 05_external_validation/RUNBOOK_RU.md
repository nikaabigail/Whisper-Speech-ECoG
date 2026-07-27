# Два внешних датасета на двух Windows-компьютерах

Этот план относится только к новому каталогу `05_external_validation`. Исторические
эксперименты `01`–`04`, их веса и исходные данные не изменяются.

## Что именно проверяем

Мы не переносим обученные пациент-специфичные веса Ивановой на других людей:
электроды, число каналов и распределения сигналов несовместимы. На каждом новом
участнике нейронный регрессор и классификационная/continuous-голова обучаются заново.
Неизменными остаются:

- замороженный multilingual `openai/whisper-base` на закреплённом commit;
- слои L3, L4 и L5;
- одинаковая 50-мерная train-only PCA-цель для MEL и Whisper;
- одно и то же лаговое ECoG-окно `[t-1000 мс, t]`; из-за zero-phase
  фильтрации весь текущий эксперимент остаётся офлайн, а не real-time causal;
- регрессия остаётся на исторической сетке 1000 Гц, а hidden-траектория для
  word-head берётся с шагом 10, то есть на 100 Гц;
- одна архитектура ECoG-энкодера и одна политика обучения;
- заранее фиксированный ансамбль: среднее softmax-вероятностей L3/L4/L5.

Ускоренная сетка 50 Гц разрешена только для технического smoke-теста. Она меняет
физическую ширину временной свёртки word-head и не может попасть в таблицу статьи.

| Компьютер | Датасет | Что он добавляет к статье |
|---|---|---|
| Текущий RTX 5070 Laptop | SingleWordProductionDutch (SWPD), 10 участников | Межпациентская проверка реконструкции акустических представлений; после независимой аудио-разметки — continuous speech detection |
| Второй, предположительно RTX 3060 Ti | VocalMind, Mandarin, 1 участник | Повторяемая 20-классовая синхронная задача и прямой тест фиксированного L3+L4+L5 ensemble |

SWPD нельзя честно превратить в обычную 100-классовую задачу: каждое из 100 слов у
участника встречается один раз. VocalMind нельзя называть свободным асинхронным
датасетом: его word-записи уже нарезаны на трёхсекундные испытания. Поэтому две
задачи дополняют друг друга, а не искусственно выдают разные постановки за одну.

## Защита от утечки

- SWPD `sub-01` — только разработка и проверка кода. `sub-02`–`sub-10` программно
  не входят в frozen v1 и останутся закрытыми до отдельного confirmatory-релиза.
- VocalMind: primary использует только repetitions 1–5. В каждом из пяти folds:
  60 trials train, 20 validation, 20 held-out test. Неполный repetition 6 — только
  development-набор для проверки форм, памяти и временного положения аудио; он не
  входит в обучение или оценку опубликованной модели.
- StandardScaler/PCA обучаются только на train. Epoch и checkpoint выбираются только
  по validation loss. Test открывается после фиксации MEL/L3/L4/L5 для данного fold.
- Seeds `1, 2, 3, 4, 42` оценивают устойчивость оптимизации и не увеличивают число
  биологических участников.

## Чистая установка второго Windows 11 компьютера

Откройте обычный PowerShell. Администратор нужен только если политика конкретного
компьютера этого потребует.

```powershell
winget install --exact --id Git.Git --accept-package-agreements --accept-source-agreements
winget install --exact --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
```

Закройте PowerShell и откройте заново, затем:

```powershell
git clone --branch codex/external-validation --single-branch https://github.com/nikaabigail/Whisper-Speech-ECoG.git C:\Whisper-Speech-ECoG
Set-Location C:\Whisper-Speech-ECoG\05_external_validation
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\inspect_host.ps1
```

Передайте вывод `inspect_host.ps1` в рабочий чат. Особенно важны точное имя GPU,
VRAM, версия NVIDIA driver, RAM и свободное место. Если `nvidia-smi` отсутствует,
сначала установите актуальный NVIDIA driver. Отдельный CUDA Toolkit не нужен.

После проверки железа:

В примерах ниже `D:` — диск артефактов. Если на втором компьютере его нет,
подставьте диск из отчёта `inspect_host.ps1`; перед production желательно иметь
не менее 60 ГиБ свободно под окружение, модельный кэш, данные и resumable-артефакты.

```powershell
.\scripts\bootstrap_windows.ps1 `
  -Dataset vocalmind `
  -DataRoot C:\WhisperECoG\VocalMind `
  -CacheRoot D:\WhisperECoG\model_cache\huggingface
```

Скрипт создаёт отдельный Python 3.10 environment, ставит закреплённый
`torch 2.10.0+cu128` и выполняет настоящий CUDA forward/backward. Данные обязательно
лежат вне Git-каталога и желательно в пути без кириллицы.

## Загрузка и строгая инвентаризация VocalMind

```powershell
.\.venv\Scripts\python.exe .\download_dataset.py `
  --profile overt_word_raw_primary `
  --destination C:\WhisperECoG\VocalMind

.\.venv\Scripts\python.exe -m whisper_ecog_ext.data.vocalmind index `
  --data-root C:\WhisperECoG\VocalMind `
  --index-out C:\WhisperECoG\VocalMind\dataset_index.json `
  --splits-out C:\WhisperECoG\VocalMind\primary_splits.json `
  --metadata-only --skip-file-hashes
```

Ожидается 119 EEG/audio-пар: 100 primary trials и 19 trials repetition 6. Каждый
primary fold обязан иметь `60/20/20`, а `rep6_primary_forbidden` — `true`.

Перед production сначала выполняется настоящий инженерный smoke на repetition 6.
Он скачивает закреплённый Whisper, проверяет preprocessing, L3/L4/L5, один
forward/backward регрессора и word-head, но не вычисляет классификационную метрику:

```powershell
.\scripts\run_vocalmind_rep6_smoke.ps1 `
  -DataRoot "C:\WhisperECoG\VocalMind" `
  -OutputRoot "D:\WhisperECoG\smoke\vocalmind_rep6_gpu" `
  -CacheRoot "D:\WhisperECoG\model_cache\huggingface"
```

Числовые данные repetitions 1–5 этим скриптом программно запрещены. Повторный
smoke следует писать в новый `OutputRoot`, чтобы первый receipt оставался
неизменным.

## Frozen production VocalMind на втором компьютере

Production запускается только из чистого checkout точного freeze-коммита. После
публикации ветки замените placeholder на SHA, сообщённый в рабочем чате:

```powershell
git checkout --detach <FREEZE_COMMIT_SHA>
git status --short
```

Пустой вывод `git status --short` обязателен. Первый ограниченный вызов обучает
одну эпоху первого каскада, сохраняет полный resumable checkpoint и нужен для
измерения времени, не открывая ни один test:

```powershell
.\scripts\start_vocalmind_production.ps1 `
  -DataRoot "C:\WhisperECoG\VocalMind" `
  -OutputRoot "D:\WhisperECoG\runs\vocalmind_frozen_v1" `
  -CacheRoot "D:\WhisperECoG\model_cache\huggingface" `
  -MaxEpochsThisCall 1
```

Скрипт напечатает путь `launcher.json`. Наблюдение:

```powershell
.\scripts\watch_background_run.ps1 `
  -LauncherReceipt "D:\WhisperECoG\runs\vocalmind_frozen_v1.launcher\launcher.json" `
  -Follow
```

`Ctrl+C` здесь останавливает только просмотр. Когда ограниченный процесс закончен
и его лог проверен, повторите первый `start_vocalmind_production.ps1` с теми же
тремя путями, но без `-MaxEpochsThisCall`. Он продолжит тот же checkpoint.
Production содержит 150 каскадов, поэтому это не «одна ночь»; реальный ETA следует
считать по времени ограниченной эпохи на конкретной RTX 3060 Ti.

## Текущий компьютер: SWPD

Данные должны находиться в `C:\WhisperECoG\SWPD`. Полная проверка авторского
MEL-контроля на `sub-01`:

```powershell
# Откройте PowerShell в каталоге 05_external_validation вашего checkout.
Set-ExecutionPolicy -Scope Process Bypass

.\.venv\Scripts\python.exe .\swpd_author_mel.py `
  --data-root C:\WhisperECoG\SWPD\extracted\SingleWordProductionDutch-iBIDS `
  --output-dir C:\WhisperECoG\SWPD\runs\author_mel_sub01_full `
  --subject sub-01 --seed 0 --randomizations 1000
```

Matched MEL80/Whisper development-анализ (не открывает других участников):

```powershell
.\.venv\Scripts\python.exe .\swpd_matched_linear.py `
  --data-root C:\WhisperECoG\SWPD\extracted\SingleWordProductionDutch-iBIDS `
  --cache-dir C:\WhisperECoG\SWPD\cache\matched_linear_sub01 `
  --output-dir C:\WhisperECoG\SWPD\runs\matched_linear_sub01 `
  --subject sub-01 --device cuda
```

Visual events здесь определяют только независимые блоки train/validation/test и не
выдаются за акустическое начало речи. Speech-only и asynchronous метрики остаются
закрытыми, пока аудио-интервалы не построены независимо и не прошли ручной аудит.

## Порядок вычислений

1. Инвентаризация файлов, checksums и воспроизведение авторского MEL-контроля.
2. SWPD: development `sub-01`; VocalMind: только rep6 engineering smoke.
3. Проверка логов, форм тензоров, отсутствия padding/утечки и разумности времени.
4. Checkout уже опубликованного frozen v1 commit и проверка чистого worktree.
5. Один неизменный VocalMind production run сразу содержит все пять folds/seeds;
   его test-gate откроется только после всех заранее объявленных каскадов.
6. SWPD `sub-02`–`sub-10` открываются отдельным confirmatory runner только после
   успешного development-аудита `sub-01`.

Не следует одновременно запускать две тяжёлые CUDA-задачи на одном GPU. Остановка
скрипта-наблюдателя через `Ctrl+C` безопасна только тогда, когда само обучение было
запущено отдельным фоновым процессом; прямой foreground-процесс `Ctrl+C` прерывает.

## Что переносить между компьютерами

Исходные EEG/audio, caches и checkpoints в Git не добавляются. Между системами
синхронизируются только один и тот же Git commit, frozen configs и небольшие итоговые
JSON/CSV/PNG с hashes. Полные рабочие каталоги остаются локальными до проверки
лицензий и правил публикации.

## Запуск полного neural-пилота SWPD sub-01

Откройте PowerShell в `05_external_validation`. Основной запуск использует
регрессионную сетку 1000 Гц, три заранее заданные инициализации MEL
(`4, 1004, 2004`) и по одной модели Whisper L3/L4/L5 (`seed=4`):

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_swpd_sub01_neural_pilot.ps1 `
  -DataRoot "C:\WhisperECoG\SWPD\extracted" `
  -CacheDir "C:\WhisperECoG_Work\SWPD\cache_1000hz" `
  -RunDir "C:\WhisperECoG_Work\SWPD\runs\seed4_v1"
```

Для ночного фонового запуска той же полной задачи:

```powershell
.\scripts\start_swpd_sub01_neural_pilot.ps1 `
  -DataRoot "C:\WhisperECoG\SWPD\extracted" `
  -CacheDir "C:\WhisperECoG_Work\SWPD\cache_1000hz" `
  -RunDir "C:\WhisperECoG_Work\SWPD\runs\seed4_v1"

.\scripts\watch_background_run.ps1 `
  -LauncherReceipt "C:\WhisperECoG_Work\SWPD\runs\seed4_v1\launcher\launcher.json" `
  -Follow
```

Скрипт сначала проверяет CUDA и создаёт отдельный TSV кандидатов речи только из
аудио. Этот TSV ещё не является разметкой. Visual cue не используется как
начало слова. Регрессионный test открывается только после фиксации всех моделей,
а асинхронный/event test остаётся закрытым до ручного прослушивания и отдельного
подписанного audit receipt.

Кэш каждого из пяти блоков имеет checksum. Checkpoint сохраняет последнюю
завершённую эпоху, best validation weights, optimizer и RNG. Если foreground
процесс остановлен посреди эпохи, повторите ровно ту же команду: завершённые
эпохи и блоки будут переиспользованы. Не меняйте `MaxEpochs`, `BatchSize`, пути
или флаги внутри одного run-каталога — fingerprint намеренно запретит такой
тихий drift.

Короткий тест разрешён только в отдельных каталогах и не является результатом:

```powershell
.\scripts\run_swpd_sub01_neural_pilot.ps1 `
  -FastSmoke `
  -CacheDir "C:\WhisperECoG_Work\SWPD\cache_smoke_50hz" `
  -RunDir "C:\WhisperECoG_Work\SWPD\runs\smoke_50hz"
```

Перед реальным запуском тесты должны завершиться без ошибок:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
