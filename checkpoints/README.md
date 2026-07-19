# Локальные веса и frozen archive

[← На главную](../README.md) · [Происхождение](../docs/PROVENANCE.md) ·
[Чек-лист публикации](../docs/PUBLISHING_CHECKLIST.md)

Обученные веса намеренно не включены в публичный кандидат. Для их публикации
нужно отдельно подтвердить:

- разрешение владельцев данных и условия информированного согласия;
- допустимость распространения моделей, обученных на пациентских данных;
- лицензию исходного кода и сторонних моделей;
- отсутствие возможности нежелательной утечки чувствительной информации;
- согласование с ответственным исследователем/организацией.

## Ожидаемая структура

Для frozen replay и continuous-head нужен самодостаточный архив upstream seed 4:

```text
checkpoints/
└── frozen_seed4/
    ├── MANIFEST.sha256.json
    └── upstream/
        ├── checkpoints/
        │   ├── L3/
        │   │   ├── regression___...WHISPER_BASE_L3...pth
        │   │   └── classification_hidden___...WHISPER_BASE_L3...pth
        │   ├── L4/
        │   │   ├── regression___...WHISPER_BASE_L4...pth
        │   │   └── classification_hidden___...WHISPER_BASE_L4...pth
        │   └── L5/
        │       ├── regression___...WHISPER_BASE_L5...pth
        │       └── classification_hidden___...WHISPER_BASE_L5...pth
        └── results/
            ├── L3/classification_hidden___...WHISPER_BASE_L3...json
            ├── L4/classification_hidden___...WHISPER_BASE_L4...json
            └── L5/classification_hidden___...WHISPER_BASE_L5...json
```

Итого: runtime manifest плюс девять payload-файлов. Точные имена, размеры,
SHA-256 и поле `expected_archive_relative_path` перечислены в
`release_manifest.json`. Сам `release_manifest.json` документирует ожидаемый
локальный набор, но не заменяет runtime-файл `MANIFEST.sha256.json`.

## Правила подключения

1. Получите веса только через разрешённый внутренний канал.
2. Поместите каждый файл по его `expected_archive_relative_path` относительно
   `checkpoints/frozen_seed4/`; не переименовывайте payload.
3. Сверьте размер и SHA-256 с release manifest.
4. Создайте `MANIFEST.sha256.json` с `seed=4` и массивом `files`, где для каждого
   payload указаны `relative_path`, `bytes` и `sha256`.
5. Не используйте файл, если хотя бы одна контрольная сумма не совпала.

Внутренний локальный архив можно собрать проверяющим скриптом, если исходные
веса уже разрешённо находятся в рабочем sync-проекте:

```powershell
.\scripts\prepare_local_checkpoint_bundle.ps1 -SourceRoot <путь-к-sync-проекту> -VerifyOnly
.\scripts\prepare_local_checkpoint_bundle.ps1 -SourceRoot <путь-к-sync-проекту>
```

Первая команда только проверяет все девять хэшей и ничего не копирует. Вторая
копирует файлы только в git-ignored локальные
каталоги и создаёт runtime manifest. Никакие исходные веса он не удаляет.

Проверка хэша в PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 `
  .\checkpoints\frozen_seed4\upstream\checkpoints\L3\regression___...WHISPER_BASE_L3...pth
```

## Что нельзя коммитить

- `.pth`, `.pt`, `.ckpt`, `.npy`, `.npz`, `.h5`, `.hdf5`;
- hidden caches и dense timelines;
- логи, содержащие абсолютные локальные пути;
- `patients.json` и любую таблицу соответствия псевдонимов данным;
- производные данные без формального решения об их статусе.

Даже если GitHub технически принимает файл по размеру, это не делает его
публикацию этически или юридически допустимой.
