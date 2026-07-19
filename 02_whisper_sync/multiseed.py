# -*- coding: utf-8 -*-
"""
Мульти-сид сравнение цели регрессии Whisper-enc-L4 vs mel для одного пациента.
Под каждым сидом прогоняет ВЕСЬ цикл regression -> hidden для обеих целей, затем
агрегирует test_accuracy (hidden) -> mean±SD и парную разницу Whisper - mel.

Запуск:   py -3.10 multiseed.py ivanova
          py -3.10 multiseed.py procenko --seeds 42 1 2
          py -3.10 multiseed.py ivanova --aggregate-only     # только пересчитать таблицу
          py -3.10 multiseed.py ivanova --target WHISPER_BASE_L3

Перед каждым сидом переносит прежние regression `.pth` именно этих моделей в
игнорируемый `artifacts/checkpoint_archive` (чтобы hidden видел ровно один
чекпойнт текущего сида). Веса не удаляются; result-JSON не трогаются, а при
агрегации дедуплицируются по сиду.
Долго (GPU): ~5 сидов × 2 модели × (regression+hidden). Лучше не на батарее.
"""
import os
import sys
import glob
import json
import argparse
import subprocess
import statistics
import shutil
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # чтобы кириллица/символы не роняли вывод в cp1251-консоли
except Exception:
    pass

PY = ["py", "-3.10", "train_sync.py"]


def model_names(patient, target):
    """Build the retained Whisper-base L3/L4/L5 and 40-MEL model names."""
    ch = "8_16" if patient == "ivanova" else "6_12"
    return {
        "whisper": f"SimpleNetBase_WithLSTM__CNANNELS_{ch}__LAG_1000_0__{target}",
        "mel":     f"SimpleNetBase_WithLSTM__CNANNELS_{ch}__LAG_1000_0__40MELS",
    }


def run(args, seed):
    env = dict(os.environ, BENCH_SEED=str(seed))
    print(f"\n>>> seed={seed}: {' '.join(args)}", flush=True)
    subprocess.run(PY + args, env=env, check=True)


def archive_pth(patient, model):
    """Keep prior encoders instead of deleting them before the next seed."""
    archive_root = Path(
        os.environ.get(
            "OSSADTCHI_CHECKPOINT_ARCHIVE",
            Path(__file__).resolve().parent.parent / "artifacts" / "checkpoint_archive",
        )
    )
    for f in glob.glob(f"model_dumps/regression___{patient}___{model}___*.pth"):
        source = Path(f)
        destination = archive_root / patient / model / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination = destination.with_name(
                f"{destination.stem}__{source.stat().st_mtime_ns}{destination.suffix}"
            )
        shutil.move(str(source), str(destination))
        print(f"[archive] {source} -> {destination}")


def aggregate(patient, models):
    """Читает hidden-JSON, дедуп по (модель, сид) по самому свежему файлу, control==none."""
    out = {}
    for tag, model in models.items():
        by_seed = {}
        for f in glob.glob(f"results/classification_hidden___{patient}___{model}___*.json"):
            if "__shuffle_labels" in f or "__aug_" in f or "__multitask" in f:
                continue
            d = json.load(open(f, encoding="utf-8"))
            if d.get("config", {}).get("control", "none") != "none":
                continue
            s = d["config"]["seed"]
            mt = os.path.getmtime(f)
            if s not in by_seed or mt > by_seed[s][1]:
                by_seed[s] = (d["test_accuracy"], mt)
        out[tag] = {s: v[0] for s, v in by_seed.items()}
    return out


def report(patient, agg):
    seeds = sorted(set(agg["whisper"]) & set(agg["mel"]))
    print("\n" + "=" * 56)
    print(f"МУЛЬТИ-СИД: {patient}  (hidden test_accuracy, chance=0.037)")
    print("=" * 56)
    print(f"{'seed':>6} | {'Whisper':>8} | {'mel':>8} | {'W-mel':>9}")
    diffs = []
    for s in seeds:
        w, m = agg["whisper"][s], agg["mel"][s]
        diffs.append(w - m)
        print(f"{s:>6} | {w:>8.3f} | {m:>8.3f} | {w - m:>+9.3f}")
    if not seeds:
        print("(нет общих сидов — сначала прогон без --aggregate-only)")
        return

    def ms(xs):
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)
    wm, ws = ms([agg["whisper"][s] for s in seeds])
    mm, msd = ms([agg["mel"][s] for s in seeds])
    dm, dsd = ms(diffs)
    print("-" * 56)
    print(f"{'mean':>6} | {wm:>8.3f} | {mm:>8.3f} | {dm:>+9.3f}")
    print(f"{'±SD':>6} | {ws:>8.3f} | {msd:>8.3f} | {dsd:>9.3f}")
    n = len(seeds)
    print(f"\nn={n} сидов. Средняя разница Whisper−mel = {dm:+.3f} ± {dsd:.3f} (SD различий).")
    if n >= 2:
        w = [agg["whisper"][s] for s in seeds]
        m = [agg["mel"][s] for s in seeds]
        wins = sum(1 for d in diffs if d > 0)
        sem = dsd / (n ** 0.5)
        print(f"Whisper > mel в {wins}/{n} сидах. SEM разницы = {sem:.3f}.")
        try:
            import scipy.stats as st
            t_p = float(st.ttest_rel(w, m).pvalue)                 # парный t-тест (правильный)
            sgn = float(st.binomtest(wins, n, 0.5, alternative="two-sided").pvalue)
            print(f"Парный t-тест p={t_p:.3f}; знаковый тест (2-стор.) p={sgn:.3f}.")
            if t_p < 0.05 and dm > 0:
                print("=> ПАРНЫЙ ТЕСТ ЗНАЧИМ (p<0.05): Whisper-цель ВЫШЕ mel.")
            elif abs(dm) <= sem:
                print("=> ПАРИТЕТ (в пределах SEM).")
            else:
                print("=> тенденция в пользу Whisper, p>=0.05 — добавь сидов.")
        except Exception as e:
            t = dm / sem if sem > 0 else float("inf")
            print(f"(scipy недоступен) t≈{t:.2f}, df={n-1}: |t|>2.78 => p<0.05 при n=5.")
    else:
        print("Для выводов нужно >=2 сидов.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patient", choices=["procenko", "ivanova"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument(
        "--target",
        choices=("WHISPER_BASE_L3", "WHISPER_BASE_L4", "WHISPER_BASE_L5"),
        default="WHISPER_BASE_L4",
        help="Whisper-base encoder layer retained in this release snapshot.",
    )
    ap.add_argument("--skip-mel", action="store_true",
                    help="не прогонять mel-базу заново — использовать уже посчитанные "
                         "mel-JSON из results/ (экономит ~половину времени).")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    target = args.target
    models = model_names(args.patient, target)
    print(f"Модели: whisper={models['whisper']}  mel={models['mel']}")

    if not args.aggregate_only:
        for seed in args.seeds:
            for tag, model in models.items():
                if tag == "mel" and args.skip_mel:
                    continue
                archive_pth(args.patient, model)
                run(["--mode", "regression", "--patient", args.patient, "--model", model, "--runs_count", "1"], seed)
                run(["--mode", "hidden", "--patient", args.patient, "--model", model], seed)

    report(args.patient, aggregate(args.patient, models))


if __name__ == "__main__":
    main()
