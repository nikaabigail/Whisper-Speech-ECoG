import argparse
import os
import os.path
import json

import library
import library.runner_regression
import library.runner_classification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--mode',
        required=True,
        choices=('regression', 'classification', 'oracle', 'hidden'),
    )
    parser.add_argument('--patient', required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument(
        '--runs_count',
        type=int,
        default=1,
        help='number of regression repetitions (default: 1)',
    )
    parser.add_argument('--control', required=False, default='none',
                        choices=['none', 'shuffle_labels'],
                        help='диагностический контроль: shuffle_labels перемешивает '
                             'метки train/val -> test должен упасть к chance (~0.037)')
    parser.add_argument('--augment', required=False, default='none',
                        choices=['none', 'specaug', 'ecog_aug'],
                        help='аугментация train-данных. specaug = аугментация кадров '
                             'КЛАССИФИКАТОРА (jitter границ + маски времени/каналов + шум). '
                             'ecog_aug = аугментация входа ЭНКОДЕРА на этапе регрессии '
                             '(шум + маски времени + channel dropout + разброс амплитуды, '
                             'без сдвига окна) — приём №1 для усиления энкодера.')
    parser.add_argument('--multitask', action='store_true', default=False,
                        help='Приём №2 (только для --mode regression): мультизадачное '
                             'обучение энкодера — он учится ОДНОВРЕМЕННО восстанавливать '
                             'мел И различать слово (доп. голова классификации + '
                             'взвешенный CrossEntropy на речевых кадрах). Делает внутреннее '
                             'представление словоразличающим. Замер потом через --mode hidden.')
    # device-agnostic: GPU используется автоматически, если доступен.
    # Чтобы выбрать конкретную карту — задайте CUDA_VISIBLE_DEVICES (необязательно).

    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()

    from library.runtime import device_str, SEED
    print(f"[runtime] device={device_str()} | seed={SEED}")

    for dir_name in ["results", "model_dumps"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            print(f"{dir_name} dir created")

    patients_dict = {}
    with open("library/patients.json", "r") as patients_file:
        for patient in json.load(patients_file):
            patients_dict[patient["name"]] = patient

    assert parsed_args.patient in patients_dict
    patient = patients_dict[parsed_args.patient]

    if parsed_args.mode == "regression":
        library.runner_regression.run_regression(
            parsed_args.model,
            patient,
            parsed_args.runs_count,
            parsed_args.debug,
            augment=parsed_args.augment,
            multitask=parsed_args.multitask
        )
    elif parsed_args.mode == "classification":
        library.runner_classification.run_classification(
            parsed_args.model,
            patient,
            parsed_args.debug,
            control=parsed_args.control,
            augment=parsed_args.augment
        )
    elif parsed_args.mode == "oracle":
        # Диагностика Фазы 1: true-mel -> word (потолок второго этапа).
        # Регрессионную модель обучать/сохранять не нужно.
        library.runner_classification.run_oracle(
            parsed_args.model,
            patient,
            parsed_args.debug,
            control=parsed_args.control,
            augment=parsed_args.augment
        )
    elif parsed_args.mode == "hidden":
        # Фаза 2.1: hidden-state -> word (внутреннее представление энкодера
        # регрессии вместо декодированной мел). Требует сохранённую модель регрессии.
        library.runner_classification.run_hidden(
            parsed_args.model,
            patient,
            parsed_args.debug,
            control=parsed_args.control,
            augment=parsed_args.augment
        )
