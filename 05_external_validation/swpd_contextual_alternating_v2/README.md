# Contextual covariance-alternating v2

Это исправленное продолжение **последнего SWPD-варианта, подготовленного для
Оссадчего**. Оно не использует старую секундную Conv/BiLSTM-ветку
`cache_1000hz_v2/seed4_v2`.

## Неизменённая база

- development-пациент `sub-01`;
- high-gamma ECoG `70–170 Гц`;
- контекст `−200…+200 мс` из девяти временных точек;
- сетка 20 мс и защитный край 1 секунда;
- пять фолдов: test `i`, validation `i+1`, остальные три блока — train;
- train-only `StandardScaler → PCA50` для neural context;
- основной декодер — OLS;
- Whisper-base L4, revision
  `e37978b90ca9030d5170a5c07aadb050351a65bb`;
- primary metric — средний Pearson `r` по общей стандартизованной MEL80-поверхности.

Cycle 0 обязан точно воспроизводить прежний результат L4:

```text
r = 0.4936210267
```

## Что исправлено относительно ошибочного v1

L4 сначала переводится в train-only whitened PCA128 search-space `H`, где

```text
Cov(H) = I128.
```

Обучаемый проектор `Q: 128→50` ограничен условием

```text
Q Qᵀ = I50.
```

Поэтому для любого допустимого `Q` все 50 targets имеют единичную train-дисперсию.
Нельзя получить ни нулевую матрицу, ни скрытый variance-collapse.

Один полный цикл:

1. при фиксированном `Q` точный OLS минимизирует
   `||decoder(X) − H Qᵀ||²`;
2. при фиксированном decoder точный rectangular Procrustes минимизирует тот же
   самый MSE;
3. код обязательно проверяет, что каждая фаза не увеличила objective;
4. endpoint цикла сравнивается с cycle 0 только на validation MEL80.

Это согласованное alternating-обучение. Здесь нет ad-hoc смешивания whitened и
unscaled losses, которое было найдено в остановленном `v1`.

## Почему здесь нет CNN/LSTM

В последнем context-matched результате основная модель — OLS. Сохранение OLS
обязательно для честного продолжения уже показанного результата. Добавление
CNN/LSTM или MLP одновременно с новым projector изменило бы сразу две причины
возможного улучшения и перестало бы быть прямым сравнением.

Термин «end-to-end» здесь означает совместное переобучение всей основной
contextual-модели и target-projector. Сам Whisper остаётся замороженным, как и в
исходном исследовании.

## Двухступенчатый запуск

Сначала запускается только fit/selection. Эта стадия не считает test-метрики:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start_fit_background.ps1
.\scripts\watch_fit.ps1 -Follow
```

После проверки `fit_summary.json` отдельной командой открывается фиксированная
test-оценка одновременно для двух ветвей:

- исходный PCA50-контроль;
- выбранный alternating v2.

```powershell
.\scripts\run_evaluate_frozen.ps1
```

Primary эффект — только парная разность
`alternating_selected − fixed_PCA50` на одинаковых кадрах. Уменьшение train MSE
само по себе результатом не считается.

По умолчанию артефакты сохраняются в:

```text
C:\WhisperECoG_Work\SWPD\runs\contextual_covariance_alternating_v2_sub01
```

Предыдущий каталог `end_to_end_alternating_l4_sub01_v1` сохраняется исключительно
как диагностическое свидетельство найденной ошибки и не должен использоваться
в публикационной таблице.
