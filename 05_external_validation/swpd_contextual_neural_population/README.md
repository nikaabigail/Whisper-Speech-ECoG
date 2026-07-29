# Frozen fixed-Q neural на SWPD sub-02…sub-09

Это population follow-up после разработки на `sub-01`. Зафиксирован победивший
вариант: neural-декодер с неизменным Whisper L4/PCA50 target-space. Alternating
исключён до population-прогона, потому что на `sub-01` он был хуже fixed-Q.

- пациенты: `sub-02…sub-09`, `n=8`; `sub-10` сохраняет прежнее QC-исключение;
- пять временных folds и seeds `1,2,3,4,42`;
- 5 циклов × 10 эпох, архитектура и optimizer без нового подбора;
- число входных каналов определяется отдельно для пациента, hidden-размеры
  остаются фиксированными;
- fit/validation не открывает fold-role test; test запускается отдельной командой.

Запуск fit-only в фоне:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_fit_background.ps1
.\scripts\watch_fit.ps1 -Follow
```

Только после `FIT COMPLETE`:

```powershell
.\scripts\run_evaluate.ps1
```

Primary статистическая единица — пациент: сначала усредняются folds внутри seed,
затем seeds внутри пациента, и только затем считается парная разница по восьми
пациентам. Это frozen secondary analysis, поскольку SWPD-когорта ранее уже
использовалась в линейном contextual-анализе.
