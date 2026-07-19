# -*- coding: utf-8 -*-
"""
Цель регрессии = эмбеддинги ЭНКОДЕРА Whisper (верхне-средний «speech»-tap слой),
выровненные кадр-в-кадр с ECoG на таймбейзе T_down (~1 кГц). Замена 40 log-mel.

Поток: sound(fs) -> 16 кГц -> 30-сек чанки -> encoder.hidden_states[layer] (~50 Гц)
       -> отбросить паддинг-хвост -> resample к ecog_size (БЕЗ off-by-one [:-1]).
Снижение размерности (z-score [-> PCA]) делается ПОФАЙЛОВО в раннере на train-сплите
(fit_reducer/apply_reducer), чтобы не было утечки. Здесь — только сырые фичи + кэш.

Подтверждено исполнением: whisper-tiny enc=4 слоя d=384, base enc=6 d=512;
hidden_states имеет L+1 тензоров (B,1500,d); 1500 кадров/30с = 50 Гц.
"""
import os
import hashlib

import numpy as np
import scipy.signal
import sklearn.preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import torch
import librosa

from .runtime import DEVICE

WHISPER_FPS = 50          # нативная частота кадров энкодера Whisper (20 мс/кадр)
WHISPER_SR = 16000        # Whisper ждёт 16 кГц
CHUNK_SEC = 30            # окно энкодера Whisper
CHUNK_SAMPLES = WHISPER_SR * CHUNK_SEC  # 480000
PCA_FIT_MAX_ROWS = 200_000  # подвыборка кадров для фита PCA/scaler (память)

_MODELS = {}              # ленивый синглтон: model_name -> (encoder, feature_extractor)
_CACHE_DIR = os.environ.get("WHISPER_CACHE_DIR",
                            os.path.join(os.path.dirname(os.path.dirname(__file__)), "whisper_cache"))


def _get_model(model_name):
    if model_name not in _MODELS:
        from transformers import WhisperModel, WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(model_name)
        enc = WhisperModel.from_pretrained(model_name).encoder.eval().to(DEVICE)
        for p in enc.parameters():
            p.requires_grad_(False)
        _MODELS[model_name] = (enc, fe)
        print(f"[whisper] loaded {model_name}: enc_layers={len(enc.layers)} d={enc.config.d_model} device={DEVICE}")
    return _MODELS[model_name]


def _cache_path(sound16k, model_name, layer):
    h = hashlib.md5()
    h.update(sound16k.tobytes())
    h.update(f"{model_name}|L{layer}|fps{WHISPER_FPS}".encode())
    return os.path.join(_CACHE_DIR, f"whisper_{h.hexdigest()}.npy")


def extract_whisper_encoder_features(sound, sampling_rate, layer, model_name):
    """sound(1-D, fs=sampling_rate) -> (N_50Hz, d) сырые фичи слоя `layer` энкодера.
    Кэшируется на диск по содержимому звука (для мульти-сида/oracle не пересчитывать)."""
    sound = np.asarray(sound, dtype="float32")
    sound16k = librosa.resample(sound, orig_sr=sampling_rate, target_sr=WHISPER_SR).astype("float32")

    cpath = _cache_path(sound16k, model_name, layer)
    if os.path.isfile(cpath):
        return np.load(cpath)

    enc, fe = _get_model(model_name)
    n_real = int(round(len(sound16k) / WHISPER_SR * WHISPER_FPS))  # реальная длина без паддинга

    feats = []
    for start in range(0, len(sound16k), CHUNK_SAMPLES):
        chunk = sound16k[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:                       # последний чанк -> паддинг нулями до 30с
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        inp = fe(chunk, sampling_rate=WHISPER_SR, return_tensors="pt").input_features.to(DEVICE)
        with torch.no_grad():
            hs = enc(inp, output_hidden_states=True).hidden_states[layer]   # (1,1500,d)
        feats.append(hs[0].float().cpu().numpy())
    feats = np.concatenate(feats, axis=0)[:n_real]            # отбросить паддинг-хвост 30-с чанка

    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.save(cpath, feats)
    return feats


def align_to_ecog(feats_50hz, ecog_size):
    """Ресэмпл по времени до РОВНО ecog_size кадров. Образец — extract_mfccs
    (scipy.signal.resample, num=out_length), БЕЗ off-by-one [:-1]."""
    out = scipy.signal.resample(feats_50hz, num=ecog_size, axis=0)
    assert out.shape[0] == ecog_size, f"{out.shape[0]} != {ecog_size}"
    return out.astype("float32")


def classic_whisper_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, layer, model_name):
    """Возвращает СЫРЫЕ Whisper-фичи (ecog_size, d). z-score/PCA — в раннере (train-only)."""
    sound = sound / (np.max(np.abs(sound)) + 1e-8)           # как classic_melspectrogram_pipeline
    feats = extract_whisper_encoder_features(sound, sampling_rate, layer, model_name)
    return align_to_ecog(feats, ecog_size)


# ---- снижение размерности цели: ФИТ ТОЛЬКО НА TRAIN (без утечки) ----
def fit_reducer(train_frames, use_pca, n_components, seed=42):
    """train_frames: (Σ T_down_train, d) — кадры цели по train-файлам.
    use_pca=True -> StandardScaler -> PCA(whiten) (компоненты единичной дисперсии);
    use_pca=False -> только StandardScaler (поканальный z-score).
    Фит на случайной подвыборке (память). Возвращает sklearn-объект с .transform."""
    X = np.asarray(train_frames, dtype="float32")
    if X.shape[0] > PCA_FIT_MAX_ROWS:
        rng = np.random.RandomState(seed)
        X = X[rng.choice(X.shape[0], PCA_FIT_MAX_ROWS, replace=False)]
    if use_pca:
        reducer = make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            PCA(n_components=n_components, whiten=True, random_state=seed),
        )
    else:
        reducer = sklearn.preprocessing.StandardScaler()
    reducer.fit(X)
    return reducer


def apply_reducer(reducer, y):
    return reducer.transform(np.asarray(y, dtype="float32")).astype("float32")
