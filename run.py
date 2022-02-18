import numpy as np
from random import shuffle

import datasets
import models
import training
import pandas as pd

NUM_EXAMPLES = 2000
NUM_EPOCHS = 5

bench_datasets = {
    "MNIST": datasets.get_mnist,
    "CIFAR10": datasets.get_cifar,
    "64x64x3": datasets.get_random,
}
bench_models = {
    "DATA PROCESSING": lambda s, c: None,
    "VGG11": lambda s, c: models.VGG(input_shape=s,
                                     n_classes=c,
                                     version=11),
    "VGG16": lambda s, c: models.VGG(input_shape=s,
                                     n_classes=c,
                                     version=16),
    "ResNet-20": lambda s, c: models.ResNet(input_shape=s,
                                            n_classes=c,
                                            version=20),
    "ResNet-56": lambda s, c: models.ResNet(input_shape=s,
                                            n_classes=c,
                                            version=56),
    "WRN16-4": lambda s, c: models.ResNet(input_shape=s,
                                          n_classes=c,
                                          version="WRN16-4"),
    "WRN16-8": lambda s, c: models.ResNet(input_shape=s,
                                          n_classes=c,
                                          version="WRN16-8"),
}
bench_batch_sizes = [32, 64, 128, 256]


def _run(model_name, ds_name, bs, msg=""):
    print(f"{msg} RUNNING {model_name} ON {ds_name} WITH BS {bs}")

    get_model = bench_models[model_name]
    tr_ds, va_ds, input_shape, n_classes = bench_datasets[ds_name](bs, bs)
    model = get_model(input_shape, n_classes)

    batches = int(np.ceil(NUM_EXAMPLES / bs))
    train_times, valid_times = training.run(
        model, tr_ds, va_ds, n_iter=batches, n_epoch=NUM_EPOCHS
    )
    train_median_time = np.median(train_times)
    valid_median_time = np.median(valid_times) if valid_times else 0.

    train_sec_per_example = train_median_time / batches / bs
    valid_sec_per_example = valid_median_time / batches / bs
    return (
        ds_name,
        bs,
        model_name,
        train_sec_per_example,
        valid_sec_per_example
    )


inputs = []
all_times = []

for ds_name in bench_datasets.keys():
    for bs in bench_batch_sizes:
        for model_name in bench_models.keys():
            inputs.append((model_name, ds_name, bs))

print("RUNNING IN RANDOM ORDER")
shuffle(inputs)
for idx, inp in enumerate(inputs):
    msg = f"{idx + 1:<4} / {len(inputs):<4}"
    all_times.append(_run(*inp, msg))

time_df = pd.DataFrame(
    all_times,
    columns=["DS", "BS", "MODEL", "TRAIN_SPE", "VALID_SPE"]
)

time_df.to_csv("results.csv")
