# -*- coding: utf-8 -*-
"""Evaluate the pre-specified synchronous Whisper L3+L4+L5 ensemble.

The script never retrains a model and never chooses a layer subset on test.  It
loads a same-seed encoder/word-head pair for each of L3, L4 and L5, evaluates all
three on the identical held-out examples, averages their softmax probabilities,
and reports the fixed three-layer result.  For the published Ivanova seed-4 run,
checkpoint filenames and SHA-256 values come from ``release_manifest.json``.
Other seeds use the deterministic newest compatible same-date pair.
"""
import os
import sys
import json
import datetime
import argparse
import hashlib
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # кириллица/символы не роняют cp1251-консоль
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SYNC_ROOT = Path(
    os.environ.get("OSSADTCHI_SYNC_ROOT", SCRIPT_DIR.parent / "02_whisper_sync")
).resolve()
if str(SYNC_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNC_ROOT))

import numpy as np
import torch
import h5py
import sklearn.preprocessing

import library.bench_models_regression as bmr
import library.bench_models_classification as bmc
from library.runtime import DEVICE, set_seed, make_split, device_str, shift_ecog_lead
from library.runner_classification import (
    predict_regression_hidden, prepare_frames, load_words_info,
    get_words_filepath, prepare_x_batch_for_net, fix_class_imbalance, HIDDEN_STRIDE,
)
from library.runner_common import WORDS_REMAP

MODEL_DUMPS = SYNC_ROOT / "model_dumps"
RESULTS = SYNC_ROOT / "results"
RELEASE_MANIFEST = REPOSITORY_ROOT / "checkpoints" / "release_manifest.json"
N_CLASSES = len(WORDS_REMAP)


def load_patient(name):
    with (SYNC_ROOT / "library" / "patients.json").open(encoding="utf-8") as f:
        for p in json.load(f):
            if p["name"] == name:
                return p
    raise SystemExit(f"нет пациента {name} в patients.json")


def model_name(patient_name, layer):
    ch = "8_16" if patient_name == "ivanova" else "6_12"
    return f"SimpleNetBase_WithLSTM__CNANNELS_{ch}__LAG_1000_0__WHISPER_BASE_L{layer}"


def date_of(pth):
    # работает и для .pth, и для .json (дата без расширения)
    return os.path.splitext(os.path.basename(pth))[0].split("___")[-1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_compatible_config(
    result_path: Path,
    patient_name: str,
    model: str,
    seed: int,
) -> dict | None:
    """Return config only for a plain, same-model, same-patient, same-seed head."""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    config = payload.get("config") or {}
    expected = {
        "mode": "classification_hidden",
        "control": "none",
        "regression_model": model,
        "patient": patient_name,
        "seed": seed,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        return None
    if config.get("augment", "none") != "none":
        return None
    return config


def release_manifest_pair(patient_name: str, layer: int, seed: int) -> dict | None:
    """Load and verify the exact pair used for the reported seed-4 result."""
    if not RELEASE_MANIFEST.is_file():
        return None
    manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("source_patient") != patient_name
        or int(manifest.get("source_seed", -1)) != seed
    ):
        return None

    entries = {
        entry.get("role"): entry
        for entry in manifest.get("files", [])
        if int(entry.get("layer", -1)) == layer
    }
    required_roles = {
        "ecog_to_whisper_encoder",
        "synchronous_word_head",
        "synchronous_result_and_split_provenance",
    }
    if set(entries) != required_roles:
        raise RuntimeError(f"Release manifest has an invalid L{layer} payload set")

    paths = {
        role: (
            RESULTS / entry["filename"]
            if role == "synchronous_result_and_split_provenance"
            else MODEL_DUMPS / entry["filename"]
        )
        for role, entry in entries.items()
    }
    for role, path in paths.items():
        entry = entries[role]
        if not path.is_file():
            raise FileNotFoundError(
                f"Exact release payload is missing: {path}. "
                "Run scripts/prepare_local_checkpoint_bundle.ps1 first."
            )
        if path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Release payload size mismatch: {path}")
        if sha256_file(path) != str(entry["sha256"]).lower():
            raise RuntimeError(f"Release payload SHA-256 mismatch: {path}")

    result_path = paths["synchronous_result_and_split_provenance"]
    model = model_name(patient_name, layer)
    config = read_compatible_config(result_path, patient_name, model, seed)
    if config is None:
        raise RuntimeError(f"Release result config is incompatible: {result_path}")
    dates = {date_of(path) for path in paths.values()}
    if len(dates) != 1:
        raise RuntimeError(f"Release L{layer} payload dates do not match: {paths}")
    return {
        "date": dates.pop(),
        "encoder": paths["ecog_to_whisper_encoder"],
        "classifier": paths["synchronous_word_head"],
        "result": result_path,
        "config": config,
        "selection": "exact release manifest (filename, size, SHA-256)",
    }


def deterministic_checkpoint_pair(patient_name: str, layer: int, seed: int) -> dict:
    """Select the lexically newest complete, compatible same-date pair."""
    model = model_name(patient_name, layer)
    pattern = f"classification_hidden___{patient_name}___{model}___*.json"
    candidates = []
    for result_path in sorted(RESULTS.glob(pattern), key=lambda path: path.name):
        config = read_compatible_config(result_path, patient_name, model, seed)
        if config is None:
            continue
        date = date_of(result_path)
        encoder = MODEL_DUMPS / f"regression___{patient_name}___{model}___{date}.pth"
        classifier = (
            MODEL_DUMPS
            / f"classification_hidden___{patient_name}___{model}___{date}.pth"
        )
        if encoder.is_file() and classifier.is_file():
            candidates.append(
                {
                    "date": date,
                    "encoder": encoder,
                    "classifier": classifier,
                    "result": result_path,
                    "config": config,
                    "selection": "newest compatible pair by timestamped filename",
                }
            )
    if not candidates:
        raise FileNotFoundError(
            f"No complete compatible L{layer} pair for patient={patient_name}, seed={seed}."
        )
    return max(candidates, key=lambda item: (item["date"], item["result"].name))


def select_checkpoint_pair(patient_name: str, layer: int, seed: int) -> dict:
    exact = release_manifest_pair(patient_name, layer, seed)
    return exact if exact is not None else deterministic_checkpoint_pair(
        patient_name, layer, seed
    )


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def classes_from_words(words_info):
    """Последовательность классов сегментов для файла ТОЛЬКО из words_info (как в prepare_frames,
    но без hidden) — нужна, чтобы воспроизвести (прожечь) тот же np.random.choice в
    fix_class_imbalance на НЕ-тестовых файлах и получить байт-в-байт headline-набор тест-файлов."""
    classes = []
    last_phrase_end = 0
    for ps, pe, phrase in words_info:
        if last_phrase_end != 0 and last_phrase_end != ps:
            classes.append(WORDS_REMAP["silent"])
        classes.append(WORDS_REMAP[phrase])
        last_phrase_end = pe
    return np.array(classes, dtype=np.int64)


def per_layer_test_probs(patient, layer, debug=False, seed=4, balanced=True):
    """Evaluate one deterministic same-seed encoder/classifier checkpoint pair."""
    mname = model_name(patient["name"], layer)
    pair = select_checkpoint_pair(patient["name"], layer, seed)
    enc_date = pair["date"]
    enc_path = pair["encoder"]
    clf_path = pair["classifier"]
    config = pair["config"]
    seed_used = int(config["seed"])
    mwl_json = config.get("max_words_length")

    bench_reg = getattr(bmr, mname)(patient)
    bench_reg.model.load_state_dict(
        torch.load(enc_path, map_location=DEVICE, weights_only=True)
    )
    bench_reg.model.eval()
    hidden_dim = bench_reg.model.final_out_features
    eff_down = bench_reg.downsampling_coef * HIDDEN_STRIDE

    clf = bmc.Mel2WordHidden(hidden_dim, patient, mname)
    clf.model.load_state_dict(
        torch.load(clf_path, map_location=DEVICE, weights_only=True)
    )
    clf.model.eval()

    files = patient["files_list"]
    test_start = patient["test_start_file_classification_index"] if not debug else 1
    _, _, test_idx = make_split(len(files), test_start)
    if debug:
        test_idx = test_idx[:1]
    test_set = set(test_idx)
    n_proc = 2 if debug else len(files)     # run_hidden в debug строит X по первым 2 файлам

    # Воспроизводим RNG run_hidden БАЙТ-в-БАЙТ: он строит X по ВСЕМ файлам по порядку, вызывая
    # fix_class_imbalance на каждом (train/val жгут np.random.choice ДО тест-файлов). Поэтому
    # на не-тестовых файлах прожигаем те же розыгрыши (классы из words_info, hidden не нужен) ->
    # подвыборка тишины тест-файлов = headline. set_seed раз перед слоем -> ещё и выровнено между слоями.
    set_seed(seed)
    classes_all, logits_all = [], []
    for idx in range(n_proc):
        words_info = load_words_info(get_words_filepath(files[idx]))
        if idx not in test_set:
            if balanced:
                cls = classes_from_words(words_info)
                fix_class_imbalance(np.empty(len(cls), dtype=object), cls)  # только прожечь RNG
            continue
        with h5py.File(files[idx], "r") as fh:
            data = fh["RawData"]["Samples"][()]
        ecog = data[:, patient["ecog_channels"]].astype("double")
        x = bench_reg.preprocess_ecog(ecog, patient["sampling_rate"]).astype("float32")
        x = shift_ecog_lead(x, getattr(bench_reg, "ECOG_LEAD_MS", 0),
                            patient["sampling_rate"] / bench_reg.downsampling_coef)  # no-op для базы
        hid = predict_regression_hidden(bench_reg, x, HIDDEN_STRIDE)
        hid = sklearn.preprocessing.scale(hid, copy=False)
        x_frames, classes = prepare_frames(hid, words_info, eff_down)
        mwl = mwl_json or max(10, int(np.percentile([xf.shape[0] for xf in x_frames], 95)))
        arr = np.empty(len(x_frames), dtype=object)  # 1-D ragged object-массив кадров (как в run_hidden)
        for i, xf in enumerate(x_frames):
            arr[i] = xf
        classes = np.array(classes, dtype=np.int64)
        if balanced:                                  # БАЛАНС тишины, как в run_hidden (headline-тест)
            arr, classes = fix_class_imbalance(arr, classes)
        xb = prepare_x_batch_for_net(arr, mwl)        # (n, 3030, mwl)
        with torch.no_grad():
            logits = clf.model(torch.FloatTensor(xb).to(DEVICE)).cpu().numpy()  # (n, 27)
        logits_all.extend(logits)
        classes_all.extend(classes)
    classes = np.array(classes_all, dtype=np.int64)
    probs = softmax(np.array(logits_all, dtype=np.float64))
    acc = float((probs.argmax(1) == classes).mean())
    surface = "баланс.тест ≈ headline" if balanced else "полный (с тишиной)"
    print(
        f"  [L{layer}] пара {enc_date} (seed {seed_used}; {pair['selection']}) "
        f"| n={len(classes)} | acc({surface})={acc:.3f}"
    )
    return dict(
        layer=layer,
        classes=classes,
        probs=probs,
        acc=acc,
        n=len(classes),
        date=enc_date,
        seed=seed_used,
        checkpoint_selection=pair["selection"],
    )


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate the pre-specified same-seed Whisper L3+L4+L5 ensemble."
    )
    ap.add_argument("patient", choices=["ivanova", "procenko"])
    ap.add_argument(
        "--layers", type=int, nargs="+", choices=(3, 4, 5), default=[3, 4, 5]
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=4,
        help="same training seed required for all three layer pairs (default: 4)",
    )
    ap.add_argument("--full", action="store_true",
                    help="мерить на ПОЛНОМ тесте с тишиной (НЕ как headline). По умолчанию — балансированный тест, как run_hidden/headline.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    args.layers = sorted(set(args.layers))
    if args.layers != [3, 4, 5]:
        ap.error("the release evaluator is intentionally fixed to --layers 3 4 5")

    set_seed(args.seed)
    print(f"[runtime] device={device_str()} | seed={args.seed}")
    patient = load_patient(args.patient)
    balanced = not args.full
    seed_tag = f"  [ОДИН СИД = {args.seed}]"
    surf_tag = "балансир.тест = headline" if balanced else "ПОЛНЫЙ тест с тишиной"
    print(f"Ансамбль слоёв {args.layers} | пациент {args.patient}{seed_tag}  [{surf_tag}]"
          + ("  [DEBUG: 1 файл, числа условны]" if args.debug else ""))

    per = [per_layer_test_probs(patient, L, args.debug, args.seed, balanced) for L in args.layers]
    if len(per) != 3:
        raise RuntimeError("fixed L3+L4+L5 evaluation requires all three checkpoint pairs")

    # выравнивание: метки и порядок слов должны совпасть между слоями (одна аудио-разметка)
    ref = per[0]["classes"]
    for p in per[1:]:
        assert p["n"] == per[0]["n"] and np.array_equal(p["classes"], ref), \
            f"рассинхрон слов: L{p['layer']} (n={p['n']}) != L{per[0]['layer']} (n={per[0]['n']})"

    layers = [p["layer"] for p in per]
    wmask = ref != 0                       # ТОЛЬКО СЛОВА (исключаем класс 'тишина'=0)
    n_words = int(wmask.sum())
    chance_w = 1.0 / max(1, len({int(c) for c in ref[wmask]}))
    def wacc(probs):                       # точность только по словам -> сравнимо с headline ~0.74
        return float((probs.argmax(1)[wmask] == ref[wmask]).mean())
    surf = "балансир. = headline-тест" if balanced else "полный с тишиной (НЕ headline)"
    print("\n" + "=" * 52)
    print(f"одиночные слои [{surf}], n={per[0]['n']} (слов={n_words}, тишины={per[0]['n']-n_words}, chance={1/N_CLASSES:.3f}):")
    for p in per:
        print(f"  L{p['layer']}: {p['acc']:.3f}" + (f"  (только слова {wacc(p['probs']):.3f})" if not balanced else ""))
    best_single = max(per, key=lambda p: p["acc"])

    fixed_probs = np.mean([item["probs"] for item in per], axis=0)
    fixed_acc = float((fixed_probs.argmax(1) == ref).mean())
    fixed_name = "L3+L4+L5"
    print("-" * 52)
    print(
        f"фиксированный ансамбль (усреднение softmax), лучший одиночный = "
        f"L{best_single['layer']} {best_single['acc']:.3f}:"
    )
    print(
        f"  {fixed_name:<12} {fixed_acc:.3f}   "
        f"(delta к лучш.одиночному {fixed_acc - best_single['acc']:+.3f})"
    )
    print("=" * 52)
    print(
        f"заранее заданный ансамбль: {fixed_name} = {fixed_acc:.3f}  "
        f"(delta {fixed_acc - best_single['acc']:+.3f})"
    )
    print("ВЕРДИКТ:", "ансамбль > лучшего слоя (слои дополняют друг друга)"
          if fixed_acc - best_single["acc"] > 0.005 else
          "ансамбль НЕ лучше лучшего слоя (слои избыточны)")
    print(f"ПРИМЕЧАНИЕ: ОДИН сид={args.seed} -> прирост ансамбля = ЧИСТЫЙ вклад слоёв (не усреднение шума сида)."
          + (" Тест БАЛАНСИРОВАННЫЙ = headline -> абсолют СРАВНИМ с 0.738." if balanced
             else " Тест ПОЛНЫЙ с тишиной -> абсолют НЕ сравнивать с 0.738, дельта честная."))

    # 'только слова' (исключая тишину) — для аудита; при balanced ~= основному числу
    full_ens_w = wacc(fixed_probs)
    word_singles = {f"L{p['layer']}": round(wacc(p["probs"]), 4) for p in per}
    if not balanced:
        print("-" * 52)
        print(f"ТОЛЬКО СЛОВА (n_слов={n_words}, chance={chance_w:.3f}) -> сравнимо с headline:")
        for p in per:
            print(f"  L{p['layer']}: {wacc(p['probs']):.3f}")
        print(f"  ансамбль {fixed_name}: {full_ens_w:.3f}")
        print("=> разрыв с 'полным' (~0.10) — это КЛАСС ТИШИНЫ; по словам ансамбль ≈ headline.")

    # --- СОХРАНЕНИЕ результата (логи теряются) ---
    out = {
        "kind": "layer_ensemble",
        "patient": args.patient,
        "layers": layers,
        "seed": args.seed,
        "selection_policy": "fixed L3+L4+L5 before test; no subset selection",
        "n_test": int(per[0]["n"]),
        "balanced_test": bool(balanced),
        "test_surface": ("balanced (= run_hidden/headline test, fix_class_imbalance)" if balanced
                         else "full unbalanced (all words+silence, NOT headline)"),
        "chance": round(1.0 / N_CLASSES, 4),
        "per_layer": [{"layer": p["layer"], "seed": p.get("seed"), "date": p["date"], "checkpoint_selection": p["checkpoint_selection"], "acc": round(p["acc"], 4)} for p in per],
        "best_single": {"layer": best_single["layer"], "acc": round(best_single["acc"], 4)},
        "fixed_ensemble": {"layers": fixed_name, "acc": round(fixed_acc, 4), "delta_vs_best_single": round(fixed_acc - best_single["acc"], 4)},
        "word_only": {"n_words": n_words, "chance": round(chance_w, 4), "per_layer": word_singles, "ensemble": round(full_ens_w, 4)},
        "note": f"single-seed={args.seed}; fixed L3+L4+L5 (no test subset selection). "
                + ("BALANCED test = identical to run_hidden/headline (fix_class_imbalance, same split/seed/metric); only difference vs single-layer runs = softmax-averaging across layers. 'acc' comparable to 0.738."
                   if balanced else
                   "FULL test incl silence -> 'acc' NOT comparable to headline 0.74; use 'word_only'."),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fn = RESULTS / f"ensemble___{args.patient}___seed_{args.seed}___L3_4_5___{ts}.json"
    with fn.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[saved] {fn}")


if __name__ == "__main__":
    main()
