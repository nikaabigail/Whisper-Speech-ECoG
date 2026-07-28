# Architecture and parity audit

Аудит выполнен относительно фактически исполнявшегося синхронного пути
`02_whisper_sync` (`models_regression.py`, `models_classification.py`,
`runner_common.py`, `runner_regression.py`, `runner_classification.py`,
`whisper_target.py`) и фиксированного ансамбля из
`03_whisper_ensemble/ensemble_layers.py`. Старые каталоги не изменялись.

## Зафиксированный основной путь

Для production и pilot выбран строгий перенос временной архитектуры:

1. Нейронный сигнал имеет частоту 1000 Гц.
2. Для каждого момента `i` строится включительное лаговое окно
   `X[i-1000:i+1]`, то есть `(channels, 1001)` без нулевого дополнения и без
   перехода через границу trial. Индексы окна не смотрят вперёд, но zero-phase
   neural preprocessing делает полный текущий анализ офлайн/некаузальным.
3. Регрессия обучается на сетке 1000 Гц. Акустические признаки Whisper/MEL
   локально интерполируются на эту сетку только по двум соседним акустическим
   кадрам.
4. ECoG-энкодер выдаёт hidden-вектор размерности `101 × 30 = 3030` для каждого
   регрессионного момента.
5. После энкодера применяется фиксированный `hidden_stride=10`. Поэтому
   классификатор получает траекторию 100 Гц, как исторический
   `HIDDEN_STRIDE=10`.
6. Для L3, L4 и L5 обучаются отдельные регрессоры и отдельные word-head.
   Фиксированный ансамбль — арифметическое среднее трёх softmax-вероятностей;
   выбор подмножеств слоёв по validation/test запрещён.

Профиль 50 Гц оставлен только для явно помеченного `fast_smoke`. Его результаты
не являются confirmatory и не сравниваются с исторической архитектурой: при
50 Гц ядро классификатора длиной 10 покрывает 200 мс вместо 100 мс, а регрессия
видит в 20 раз меньше обучающих окон.

## Сопоставление формул

| Узел | Исторический путь 02/03 | Новый путь 05 | Статус |
|---|---|---|---|
| Нейронное окно | `X[i-1000:i+1]`, `(B,C,1001)` | То же окно внутри одного trial | **Сохранено**; устранены окна через границы файлов |
| ECoG SimpleNet | `Conv1x1 → BN(affine=False) → depthwise Conv25 → BN(affine=False) → abs → depthwise Conv15 → stride10 → BiLSTM(15×2) → flatten → BN(affine=False) → Linear` | Те же bias, groups, padding и размерности | **Численно эквивалентно** после переноса одних весов |
| Hidden ECoG | `101 × 30 = 3030` | `101 × 30 = 3030` | **Сохранено** |
| Word head | `Conv2d(1→100, kernel=(3030,10)) → MaxPool1d(10) → BiLSTM(100×2) → Linear` | Та же топология; эффективный LSTM dropout равен нулю при одном слое | **Численно эквивалентно** после переноса одних весов |
| Временная сетка word head | Регрессия около 1000 Гц, затем `HIDDEN_STRIDE=10`, то есть hidden 100 Гц | Production/pilot: 1000 Гц, затем stride 10, то есть 100 Гц | **Сохранено** |
| Whisper tap | `hidden_states[layer]`; слои считались отдельными вызовами | L3/L4/L5 из одного encoder-forward | **Сохранено и стабилизировано** |
| Whisper provenance | Плавающая версия `openai/whisper-base` | Обязательный 40-символьный commit SHA | **Усилено** для воспроизводимости |
| Аудиоамплитуда | Peak-normalization до Whisper/MEL | Безопасная peak-normalization; тишина остаётся нулём | **Сохранено** |
| Target reducer | Train-only `StandardScaler → PCA50(whiten=True)` | Та же формула, immutable artifact и checksums | **Сохранено**; расчёт в float64 даёт лишь малую численную разницу |
| Target alignment | Глобальный FFT-resample всего файла примерно до 1 кГц | Локальная интерполяция двух соседних кадров; edge-hold максимум половина кадра | **Преднамеренно исправлено**: нет зависимости от удалённых/будущих кадров |
| Neural scaling | `sklearn.scale` отдельно на полном файле, включая test | Dataset preprocessing + channel scaler, fitted только на train trials | **Преднамеренно исправлено**: test statistics не используются |
| Реализация scaling | Масштабирование полного файла до генерации окон | Train-fitted scaler один раз преобразует каждый полный trial в read-only float32; окна остаются ленивыми | **Численно идентично per-window**, но без повторного пересчёта перекрывающихся 1001-точечных окон |
| Split | Ранние файлы train, последний pre-test файл validation, поздние test | Immutable grouped dataset-specific split с disjoint IDs и checksum | **Адаптировано** к единице независимости внешнего dataset |
| Test access | Test-массивы могли создаваться до обучения | Числовой test открывается после completion receipts; evaluator требует gate authorization и точное совпадение test IDs/order | **Усилено** |
| Ensemble | Среднее softmax L3/L4/L5 | То же среднее при одинаковых sample IDs и порядке | **Сохранено и проверяется** |
| Resume | Обычно сохранялись только best weights | Atomic last+best model, Adam state, Python/NumPy/Torch/CUDA RNG и config fingerprint | **Усилено**; продолжение воспроизводит тот же run |
| Model selection | Регрессия: sampled/smoothed validation correlation; classifier: sampled validation accuracy | Полный deterministic validation loss (MSE/CE), test не используется | **Преднамеренно изменено** и должно быть описано в Methods |
| MEL target | Исторический baseline: 40 bands, 0–2 кГц | Основной внешний matched target: 80 bands, 0–8 кГц, затем train-only PCA50 | **Преднамеренно изменено** по frozen external protocol; historical profile должен запускаться отдельно |

## Проверки, которые блокируют тихий drift

- Численная эквивалентность нового ECoG-энкодера и фактического старого
  `SimpleNet` на одних весах: сравниваются hidden 3030D и regression output.
- Численная эквивалентность нового word-head и фактического старого
  `Mel2WordHidden` на одних весах.
- Точное соответствие окна `lag=-1000 ms`: старт, конец и все 1001 значения.
- Точное равенство результатов whole-trial pretransform и эталонного
  per-window scaling; storage receipt гарантирует, что перекрывающиеся окна не
  материализованы в памяти.
- Один peak-normalized Whisper-forward выдаёт L3/L4/L5, а provenance содержит
  pinned revision.
- PCA/scaler обучаются только на train IDs; test-gate сверяет split fingerprint,
  полный список обязательных units, completion fingerprints и test ID order.
- Resume проверяется против непрерывного запуска, включая caller-supplied
  initialization.

## Severity audit

| Severity | Находка | Состояние |
|---|---|---|
| HIGH | Held-out evaluator можно было вызвать без открытого gate | **Исправлено**: authorization + split/sample-ID verification |
| HIGH | 50 Гц меняло физический receptive field word-head и число regression-окон | **Исправлено**: production/pilot зафиксированы как 1000 Гц → stride 10 → 100 Гц; 50 Гц только smoke |
| HIGH | Наивный per-window scaler повторял преобразование почти одинаковых 1001-точечных окон и делал полный 1 кГц run практически непроходимым | **Исправлено**: каждый trial преобразуется один раз, окна ленивые, добавлен identity/memory test |
| MEDIUM | Existing gate receipt можно было переиспользовать с сокращённым списком required units | **Исправлено**: повторная сверка units и completion fingerprints |
| MEDIUM | Peak-normalization исторического target path отсутствовала | **Исправлено**, добавлен amplitude-invariance test |
| MEDIUM | Resume с caller-supplied initialization зависел от случайных весов нового процесса | **Исправлено**: initialization receipt читается из checkpoint |
| LOW | Локальное выравнивание могло неявно держать край на произвольном расстоянии | **Исправлено**: максимум половина target frame |

## Оставшиеся методические различия

Внешний протокол меняет learning rate, batch size, epoch-based patience и
критерий выбора checkpoint относительно исторических 400 тысяч стохастических
шагов. Это не tensor-shape bug, но эти параметры должны быть заморожены до
confirmatory run и явно перечислены в Methods. Нельзя интерпретировать разницу
результатов как эффект только Whisper, если одновременно меняются MEL profile,
split, preprocessing или model-selection rule; для такого утверждения требуется
matched ablation внутри одного и того же frozen protocol.
