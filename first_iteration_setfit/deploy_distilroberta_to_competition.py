"""This version must be tested, it was never executed, since first version of
models was created manually.

"""
import os
import shutil
import subprocess

from utils import attributes

models_dir = "/data/feedback-prize/models/"
kaggle_datasets_dir = "/data/feedback-prize/kaggle-datasets/distilroberta-models"
model_arch = "distilroberta"
models_to_deploy = []
for model_name in models_to_deploy:
    print("Deploying model: ", model_name)
    model_class = model_name.split("_")[0]
    model_path = os.path.join(models_dir, model_name)
    dataset_path = os.path.join(kaggle_datasets_dir, f"{model_class}-{model_arch}")

    # Safety checks for correct paths
    assert model_class in attributes
    assert model_class in model_path
    assert model_class in dataset_path

    pytorch_origin_path = f"{model_path}/pytorch_model.bin"
    pytorch_destiny_path = f"{dataset_path}/pytorch_model.bin"

    print("Copying {} to {}".format(pytorch_origin_path, pytorch_destiny_path))

    shutil.copy(pytorch_origin_path, pytorch_destiny_path)

    head_origin_path = f"{model_path}/model_head.pkl"
    head_destiny_path = f"{dataset_path}/model_head.pkl"

    print("Copying {} to {}".format(head_origin_path, head_destiny_path))
    shutil.copy(head_origin_path, head_destiny_path)
    with open(f"{dataset_path}/pytorch_model.bin.md5", "w") as outfile:
        subprocess.run(
            ["md5sum", pytorch_destiny_path],
            stdout=outfile,
        )

    with open(f"{dataset_path}/model_head.pkl.md5", "w") as outfile:
        subprocess.run(
            ["md5sum", head_destiny_path],
            stdout=outfile,
        )

    with open(f"{dataset_path}/model_name.txt", "w") as outfile:
        outfile.write(model_name)

subprocess.run(
    [
        "kaggle",
        "datasets",
        "version",
        "-p",
        kaggle_datasets_dir,
        "-m",
        model_name,
        "--dir-mode",
        "tar",
    ]
)
