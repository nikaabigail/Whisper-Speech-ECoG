import json
import datetime
import h5py

import numpy as np
import torch
import torch.nn as nn

from .runner_common import data_generator, WORDS_REMAP
from . import bench_models_regression
from . import whisper_target
from .runtime import DEVICE, SEED, set_seed, make_split, device_str


MAX_ITERATIONS_COUNT = 400_000
METRIC_ITERATIONS = 1_000
EARLY_STOP_STEPS = 10_000
MULTITASK_WORD_WEIGHT = 0.5  # вес word-лосса относительно mel-MSE в приёме №2 (мультизадача)


def corr_multiple(x, y):
    assert x.shape[1] == y.shape[1]
    return [np.corrcoef(x[:, i],  y[:, i], rowvar=False)[0, 1] for i in range(x.shape[1])]


def augment_ecog_batch(x_batch, noise_std=0.1, n_time_masks=2, time_mask_frac=0.10,
                       chan_drop_prob=0.3, max_chan_drop=1, amp_jitter=0.1):
    """
    Аугментация ВХОДА энкодера (ECoG-окон) на этапе регрессии (приём №1, "больше
    примеров из тех же данных"). x_batch: (B, C, T) — C каналов ЭКоГ, T отсчётов окна.
    Применяется ТОЛЬКО к train-батчам.

    ВАЖНО: цель регрессии — mel в ЦЕНТРЕ окна, поэтому глобального сдвига окна по
    времени здесь НЕТ (он рассогласовал бы вход и цель). Все операции сохраняют
    выравнивание вход<->цель:
      - гауссов шум (сигнал z-нормирован, std~1);
      - маски по времени (зануляем небольшие куски контекста — как dropout во времени);
      - channel dropout (зануляем 0..max_chan_drop электродов) — против переоплаты
        на отдельный контакт (важно: у нас всего 6-8 каналов, поэтому мягко);
      - небольшой разброс амплитуды по каналам.
    """
    B, C, T = x_batch.shape
    out = x_batch.copy()
    for i in range(B):
        for _ in range(n_time_masks):
            w = np.random.randint(0, max(1, int(T * time_mask_frac)) + 1)
            if w > 0 and T - w > 0:
                t0 = np.random.randint(0, T - w)
                out[i, :, t0:t0 + w] = 0
        if max_chan_drop > 0 and np.random.rand() < chan_drop_prob:
            k = np.random.randint(1, max_chan_drop + 1)
            chans = np.random.choice(C, size=min(k, C), replace=False)
            out[i, chans, :] = 0
        if amp_jitter > 0:
            scale = (1.0 + np.random.uniform(-amp_jitter, amp_jitter, size=(C, 1))).astype(out.dtype)
            out[i] = out[i] * scale
    if noise_std > 0:
        out = out + np.random.normal(0, noise_std, out.shape).astype(out.dtype)
    return out


def build_frame_word_labels(words_info, n_frames, downsampling_coef):
    """Пер-кадровые метки слова (0=silent) на таймбейзе мел/ЭКоГ (T_down),
    выровненные с целью регрессии. Границы из *_words.txt (в сырых отсчётах)
    переводятся в кадры делением на downsampling_coef — как в prepare_frames."""
    w = np.zeros(n_frames, dtype=np.int64)
    for phrase_start, phrase_end, phrase in words_info:
        a = max(0, int(phrase_start / downsampling_coef))
        b = min(n_frames, int(phrase_end / downsampling_coef))
        if b > a:
            w[a:b] = WORDS_REMAP[phrase]
    return w


def compute_class_weights(w_labels, n_classes=None):
    """Веса классов для CrossEntropy: обратный корень частоты, среднее=1.
    Поднимает редкие слова относительно частой тишины, не обнуляя её."""
    if n_classes is None:
        n_classes = len(WORDS_REMAP)
    counts = np.bincount(w_labels, minlength=n_classes).astype(np.float64)
    weights = 1.0 / np.sqrt(counts + 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def process_batch(bench_model, generator, is_train, iteration, augment="none",
                  aux_head=None, word_weight=0.0, class_weights=None):
    loss_function = nn.MSELoss()

    if is_train:
        bench_model.model.train()
        if aux_head is not None:
            aux_head.train()
    else:
        bench_model.model.eval()
        if aux_head is not None:
            aux_head.eval()

    batch = next(generator)
    if aux_head is not None:
        x_batch, y_batch, w_batch = batch
    else:
        x_batch, y_batch = batch

    if is_train and augment != "none":
        x_batch = augment_ecog_batch(x_batch)

    y_batch_speech_indexes = bench_model.detect_voice(y_batch)

    assert x_batch.shape[0] == y_batch.shape[0]
    x_batch = torch.FloatTensor(x_batch).to(DEVICE)
    y_batch = torch.FloatTensor(y_batch).to(DEVICE)

    if is_train:
        bench_model.optimizer.zero_grad()

    if aux_head is not None:
        # один проход энкодера -> и мел (через fc_layer), и логиты слова (доп. голова)
        hidden = bench_model.model(x_batch, return_hidden=True)
        y_predicted = bench_model.model.fc_layer(hidden)
        word_logits = aux_head(hidden)
    else:
        y_predicted = bench_model.model(x_batch)
    assert not torch.any(torch.isnan(y_predicted))

    loss = loss_function(y_predicted, y_batch)

    word_loss_value = None
    word_acc_speech = None
    if aux_head is not None:
        w_t = torch.as_tensor(w_batch, dtype=torch.long, device=DEVICE)
        ce = nn.CrossEntropyLoss(weight=class_weights)
        word_loss = ce(word_logits, w_t)
        loss = loss + word_weight * word_loss
        word_loss_value = float(word_loss.cpu().detach().numpy())
        pred_w = word_logits.argmax(dim=1).cpu().detach().numpy()
        speech_mask = w_batch != 0
        if speech_mask.any():
            word_acc_speech = float((pred_w[speech_mask] == w_batch[speech_mask]).mean())

    if is_train:
        loss.backward()
        bench_model.optimizer.step()

    assert y_predicted.shape[0] == y_batch.shape[0], f"{y_predicted.shape[0]} != {y_batch.shape[0]}"
    assert y_predicted.shape[1] == y_batch.shape[1], f"{y_predicted.shape[1]} != {y_batch.shape[1]}"

    metrics = {}

    y_predicted_numpy = y_predicted.cpu().detach().numpy()
    y_batch_numpy = y_batch.cpu().detach().numpy()

    metrics["loss"] = float(loss.cpu().detach().numpy())

    metrics["correlation"] = float(np.nanmean(corr_multiple(y_predicted_numpy, y_batch_numpy)))

    if np.any(y_batch_speech_indexes):
        metrics["correlation_speech"] = float(np.nanmean(corr_multiple(y_predicted_numpy[y_batch_speech_indexes], y_batch_numpy[y_batch_speech_indexes])))

    if word_loss_value is not None:
        metrics["word_loss"] = word_loss_value
    if word_acc_speech is not None:
        metrics["word_accuracy_speech"] = word_acc_speech

    for key, value in metrics.items():
        bench_model.logger.add_value(key, is_train, value, iteration)

    return metrics


def get_random_predictions(model, generator, iterations):
    Y_batch = []
    Y_predicted = []
    for index, batch in enumerate(generator):
        x_batch, y_batch = batch[0], batch[1]   # терпит и (x,y), и (x,y,w) из мультизадачного генератора
        x_batch = torch.FloatTensor(x_batch).to(DEVICE)
        y_predicted = model(x_batch).cpu().detach().numpy()
        assert x_batch.shape[0]==y_predicted.shape[0]
        Y_predicted.append(y_predicted)
        Y_batch.append(y_batch)
        if index > iterations:
            break

    Y_predicted = np.concatenate(Y_predicted, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    return Y_batch, Y_predicted


def run_regression(bench_model_name, patient, runs_count=1, is_debug=False, augment="none", multitask=False):
    assert hasattr(bench_models_regression, bench_model_name), f"No such model:{bench_model_name}"
    set_seed(SEED)
    print(f"[runtime] device={device_str()} | seed={SEED}" + (" | MULTITASK (mel + word)" if multitask else ""))
    bench_model = getattr(bench_models_regression, bench_model_name)(patient)

    if multitask:
        # пер-кадровые метки слова нужны только для мультизадачи (приём №2)
        from .runner_classification import load_words_info, get_words_filepath

    X = []
    Y = []
    W = [] if multitask else None

    for filepath in patient["files_list"]:
        # The training pipeline only reads source recordings.  Opening them in
        # read-only mode prevents an accidental metadata write to research data.
        with h5py.File(filepath, 'r') as input_file:
            data = input_file['RawData']['Samples'][()]

        ecog = data[:, patient["ecog_channels"]].astype("double")
        sound = data[:, patient["sound_channel"]].astype("double")

        x = bench_model.preprocess_ecog(ecog, patient["sampling_rate"]).astype("float32")
        from .runtime import shift_ecog_lead   # нейро-lead тест (no-op при ECOG_LEAD_MS=0)
        x = shift_ecog_lead(x, getattr(bench_model, "ECOG_LEAD_MS", 0),
                            patient["sampling_rate"] / bench_model.downsampling_coef)
        y = bench_model.preprocess_sound(sound, patient["sampling_rate"], x.shape[0]).astype("float32")

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        assert x.shape[0] == y.shape[0]

        X.append(x)
        Y.append(y)

        if multitask:
            words_info = load_words_info(get_words_filepath(filepath))
            W.append(build_frame_word_labels(words_info, x.shape[0], bench_model.downsampling_coef))

        if is_debug and len(X) >= 2:
            break

    test_start_file_index = bench_model.TEST_START_FILE_INDEX if not is_debug else 1

    # Честный сплит: test = отложенные файлы (только отчёт), val = последний train-файл
    # (для early stopping), train = остальные. Убирает баг val == test.
    train_idx, val_idx, test_idx = make_split(len(X), test_start_file_index)
    print(f"[split] train_files={train_idx} val_files={val_idx} test_files={test_idx}")

    # Whisper-цель: снижение размерности (z-score [-> PCA]) — ФИТ ТОЛЬКО НА TRAIN (без утечки).
    # Фитим на подвыборке кадров train-файлов (без гигантского concat), затем трансформируем ВСЕ Y.
    if getattr(bench_model, "IS_WHISPER_TARGET", False):
        rng = np.random.RandomState(SEED)
        per = max(1, whisper_target.PCA_FIT_MAX_ROWS // max(1, len(train_idx)))
        sample = np.concatenate(
            [Y[i][rng.choice(Y[i].shape[0], min(per, Y[i].shape[0]), replace=False)] for i in train_idx],
            axis=0)
        reducer = whisper_target.fit_reducer(sample, bench_model.USE_PCA, bench_model.PCA_COMPONENTS, seed=SEED)
        Y = [whisper_target.apply_reducer(reducer, y) for y in Y]
        assert Y[0].shape[1] == bench_model.OUTPUT_SIZE, f"reduced {Y[0].shape[1]} != OUTPUT_SIZE {bench_model.OUTPUT_SIZE}"
        print(f"[whisper] target reduced -> {Y[0].shape[1]} dims "
              f"(use_pca={bench_model.USE_PCA}, model={bench_model.WHISPER_MODEL}, layer={bench_model.WHISPER_LAYER})")

    X_train = np.concatenate([X[i] for i in train_idx], axis=0)
    Y_train = np.concatenate([Y[i] for i in train_idx], axis=0)

    X_val = np.concatenate([X[i] for i in val_idx], axis=0)
    Y_val = np.concatenate([Y[i] for i in val_idx], axis=0)

    X_test = np.concatenate([X[i] for i in test_idx], axis=0)
    Y_test = np.concatenate([Y[i] for i in test_idx], axis=0)

    W_train = W_val = None
    class_weights = None
    if multitask:
        W_train = np.concatenate([W[i] for i in train_idx], axis=0)
        W_val = np.concatenate([W[i] for i in val_idx], axis=0)
        class_weights = compute_class_weights(W_train).to(DEVICE)
        print(f"[multitask] кадров-речи в train: {int((W_train != 0).sum())}/{len(W_train)} "
              f"({100 * (W_train != 0).mean():.1f}%), классов={len(WORDS_REMAP)}, word_weight={MULTITASK_WORD_WEIGHT}")

    batch_size = bench_model.BATCH_SIZE
    lag_backward = bench_model.LAG_BACKWARD
    lag_forward = bench_model.LAG_FORWARD

    train_generator = data_generator(X_train, Y_train, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True, W=W_train)
    val_generator = data_generator(X_val, Y_val, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True, W=W_val)

    max_iterations_count = MAX_ITERATIONS_COUNT if not is_debug else 1_000


    for run_iteration in range(runs_count):
        print("Run iteration:", run_iteration)
        best_iteration = 0
        max_metric = -float("inf")
        max_metric_speech = -float("inf")
        bench_model = getattr(bench_models_regression, bench_model_name)(patient)

        # Приём №2: доп. голова классификации поверх внутреннего представления
        # энкодера (features_scaled). Учится вместе с энкодером, на инференсе не нужна
        # (как и проектор в CLIP) — нам важно, чтобы СЛОВОРАЗЛИЧАЮЩИМ стал энкодер.
        aux_head = None
        mt_kwargs = {}
        if multitask:
            aux_head = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(bench_model.model.final_out_features, len(WORDS_REMAP)),
            ).to(DEVICE)
            bench_model.optimizer.add_param_group({"params": list(aux_head.parameters())})
            mt_kwargs = dict(aux_head=aux_head, word_weight=MULTITASK_WORD_WEIGHT, class_weights=class_weights)

        # Суффиксы __aug_/__multitask ставятся ПОСЛЕ временной метки (после '___'),
        # поэтому каскадный поиск регрессионной модели (split по '___') не ломается.
        aug_suffix = f"__aug_{augment}" if augment != "none" else ""
        mt_suffix = "__multitask" if multitask else ""
        model_filename = f"regression___{patient['name']}___{bench_model.__class__.__name__}___{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{aug_suffix}{mt_suffix}"
        model_path =  f"model_dumps/{model_filename}.pth"
        for iteration in range(max_iterations_count):
            process_batch(bench_model, train_generator, True, iteration, augment=augment, **mt_kwargs)
            with torch.no_grad():
                metrics = process_batch(bench_model, val_generator, False, iteration, **mt_kwargs)
                is_last_iteration = iteration == (max_iterations_count - 1)
                if iteration % 1000 == 0 or is_last_iteration:
                    if multitask:
                        # отбор по val-точности РАЗЛИЧЕНИЯ СЛОВ (на речевых кадрах):
                        # цель приёма №2 — словоразличающий энкодер, а не мел-корреляция
                        smoothed_word = bench_model.logger.get_smoothed_value("word_accuracy_speech")
                        if smoothed_word >= max_metric:
                            max_metric = smoothed_word
                            best_iteration = iteration
                            torch.save(bench_model.model.state_dict(), model_path)
                        else:
                            assert iteration >= best_iteration
                            if (iteration - best_iteration) > EARLY_STOP_STEPS:
                                print(f"Stopping model (multitask). Iteration {iteration} word_acc_speech={round(smoothed_word, 3)}. Best iteration {best_iteration} {round(max_metric, 3)}.")
                                break
                    else:
                        smoothed_metric = bench_model.logger.get_smoothed_value("correlation")
                        smoothed_metric_speech = bench_model.logger.get_smoothed_value("correlation_speech")
                        if smoothed_metric >= max_metric or smoothed_metric_speech >= max_metric_speech:
                            max_metric = max(smoothed_metric, max_metric)
                            max_metric_speech = max(smoothed_metric_speech, max_metric_speech)
                            best_iteration = iteration
                            torch.save(bench_model.model.state_dict(), model_path)
                        else:
                            assert iteration >= best_iteration
                            if (iteration - best_iteration) > EARLY_STOP_STEPS:
                                print(f"Stopping model. Iteration {iteration} {round(smoothed_metric, 2)} {round(smoothed_metric_speech, 2)}. Best iteration {best_iteration} {round(max_metric, 2)} {round(max_metric_speech, 2)}.")
                                break
        bench_model.model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True)
        )
        bench_model.model.eval()

        test_generator = data_generator(X_test, Y_test, batch_size, lag_backward, lag_forward, shuffle=True, infinite=True)

        result = {}
        result["train_corr"] = np.mean(corr_multiple(*get_random_predictions(bench_model.model, train_generator, METRIC_ITERATIONS)))
        result["val_corr"] = np.mean(corr_multiple(*get_random_predictions(bench_model.model, val_generator, METRIC_ITERATIONS)))
        result["test_corr"] = np.mean(corr_multiple(*get_random_predictions(bench_model.model, test_generator, METRIC_ITERATIONS)))
        result["train_logs"] = bench_model.logger.train_logs
        result["val_logs"] = bench_model.logger.test_logs
        result["iterations"] = iteration
        result["config"] = {
            "mode": "regression",
            "model": bench_model_name,
            "patient": patient["name"],
            "seed": SEED,
            "device": str(DEVICE),
            "split": {"train_files": train_idx, "val_files": val_idx, "test_files": test_idx},
            "best_iteration": best_iteration,
            "augment": augment,
            "multitask": multitask,
            "word_weight": MULTITASK_WORD_WEIGHT if multitask else None,
            "selection_metric": "word_accuracy_speech" if multitask else "correlation",
        }

        with open(f'results/{model_filename}.json', 'w') as result_file:
            json.dump(result, result_file)
