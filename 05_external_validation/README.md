# Внешняя валидация: SWPD и VocalMind

Отдельный трек внешней валидации, не изменяющий исторические пайплайны
Ivanova/Procenko и не использующий их patient-specific веса.

[SWPD frozen contextual — финальный результат](swpd_contextual_frozen/README.md) ·
[SWPD neural end-to-end — development fit](swpd_contextual_neural_e2e/README.md) ·
[SWPD linear alternating — сохранённый контроль](swpd_contextual_alternating_v2/results/README.md) ·
[SWPD matched PCA50 — предварительный target-space анализ](swpd_matched_pca50/README.md) ·
[SWPD setup](SWPD_README.md) ·
[VocalMind runbook](VOCALMIND_PRIMARY_RUNBOOK.md) ·
[Общий протокол](PROTOCOL_DRAFT.md) ·
[Лабораторный журнал](LAB_JOURNAL.md)

## Текущий статус

| Dataset | Постановка | Статус |
|---|---|---|
| SWPD final | Frozen contextual MEL80 vs заранее выбранный Whisper L4/PCA50 на общей MEL80-поверхности | **Завершено: primary n=8** |
| SWPD development | `sub-01`: PCA50 против sRRR50, CLIP50 и alternating50 | **Завершено: выбран L4/PCA50; альтернативы отклонены** |
| SWPD neural follow-up | `sub-01`: paired fixed-PCA50 neural против neural alternating, 5 folds × 5 seeds | Код готов; test закрыт до полного fit-only |
| VocalMind | Повторяемое Mandarin word decoding на второй системе | Выполняется отдельно; результаты этого компьютера не подменяют внешний run |

## SWPD: финальный frozen contextual-результат

![SWPD frozen contextual result](swpd_contextual_frozen/figures/figure_01_frozen_main.png)

| Система | Средний Pearson `r` | 95% t-CI |
|---|---:|---:|
| Прямой MEL80 | 0,69187 | [0,60060; 0,78314] |
| Whisper L4 → PCA50 → MEL80 | **0,69290** | [0,60190; 0,78390] |
| Δ L4−MEL | **+0,00103** | **[+0,00002; +0,00204]** |

L4 превысил MEL80 у `6/8` пациентов. Предзаданный paired t-test: `p=0,0462`;
точный знаковый тест: `p=0,2891`. Эффект положительный, но очень мал и требует
осторожной интерпретации.

Полный разбор: [swpd_contextual_frozen](swpd_contextual_frozen/README.md).

## Scientific guardrails

- SWPD имеет 100 уникальных слов на участника и не используется как обычная
  within-subject 100-class классификация.
- Visual cue timestamps задают только границы temporal blocks и не называются
  акустическими началами слов.
- Primary SWPD metric — all-frame representation predictability, не word accuracy.
- High-gamma и Whisper targets офлайн/некаузальны; SWPD-результат не является
  real-time или asynchronous decoder.
- Финальная система L4 была выбрана только на `sub-01`; L3/L5 и learned
  bottlenecks не перебирались на confirmatory cohort.
- Patient, а не frame, fold или optimizer seed, является статистической единицей.
- `sub-01` исключён из primary inference как development subject.
- `sub-10` исключён по source-only QC: официальная запись заканчивается после
  95 валидных trials. Решение, контрольные суммы и отсутствие модели для него
  сохранены отдельно.

## Что публикуется

- read-only adapters и код extraction/training/evaluation;
- frozen configs и QC amendment;
- агрегированные JSON/CSV и PNG/SVG;
- машинно-проверяемые receipts и аудит;
- Windows 11 bootstrap/run scripts.

Не публикуются исходные записи, производные массивы, caches, checkpoints, логи,
локальные пути к данным и чувствительные метаданные.

## Windows 11 setup

После клонирования откройте PowerShell в этой папке.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1 -Dataset swpd -DataRoot "C:\WhisperECoG\data"
```

Для VocalMind на отдельной системе:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1 -Dataset vocalmind -DataRoot "D:\WhisperECoG\data" -InstallSystemTools
```

Точный порядок подготовки данных и запуска описан в [RUNBOOK_RU.md](RUNBOOK_RU.md).
Разработческие решения и обнаруженные дефекты записаны в
[LAB_JOURNAL.md](LAB_JOURNAL.md).
