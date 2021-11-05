import warnings

warnings.filterwarnings("ignore")

import json
import os

import torch
from PIL import Image
from tqdm import tqdm

DATASET = os.path.join("data", "osld")
BASE_DIR = os.path.join(DATASET, "product-images")
ANNO_DIR = os.path.join(DATASET, "annotations")
# OUTP_DIR = os.path.join(DATASET, "product-images-cropped")

# Init directories
# def prepare_dataset(data_type, how="logo"):
def prepare_dataset(data_path, data_type, data_type_crop, how="logo"):

    if not os.path.exists("{}/{}".format(data_path, data_type_crop)):
        os.mkdir("{}/{}".format(data_path, data_type_crop))

    train_images = {}
    # for img_id, img_name in tqdm(
    #     images.items(), desc="process {} data for cub dataset".format(data_type)
    # ):

    # out_dir = OUTP_DIR + f"-by-{how}"

    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)

    # dataset_path = os.path.join(out_dir, data_type)

    # if not os.path.exists(dataset_path):
    #     os.mkdir(dataset_path)

    anno_fn = f"osld-{data_type}.json"
    with open(os.path.join(ANNO_DIR, anno_fn)) as f:
        data = json.load(f)

    for img_no, img_fn in enumerate(data, 1):
        for bbox_no, anno in enumerate(data[img_fn], 1):
            bbox, label = anno
            if how == "logo":
                label = label.split(".")[0] if not label.startswith("__") else "unknown"
            elif how == "brand":
                label = label.split("-")[0] if not label.startswith("__") else "unknown"

            # out_label_dir = os.path.join(dataset_path, label)
            # if not os.path.exists(out_label_dir):
            #     os.mkdir(out_label_dir)

            if data_type == "uncropped":
                img = Image.open(os.path.join(BASE_DIR, img_fn)).convert("RGB")

            else:
                img = Image.open(os.path.join(BASE_DIR, img_fn))
                left, top, right, bottom = bbox
                img = img.crop((left, top, right, bottom))

            save_name = os.path.join(data_path, data_type, os.path.basename(img_fn))
            img.save(save_name)

            # if int(label[img_no]) < 101:
            if label[img_no] in train_images:
                train_images[label].append(save_name)
            else:
                train_images[label] = [save_name]

            # else:
            #     if label[img_no] in test_images:
            #         test_images[label[img_no]].append(save_name)
            #     else:
            #         test_images[label[img_no]] = [save_name]
        # print(f"finished cropping image {img_no} - {img_fn}")

        return train_images


if __name__ == "__main__":
    # for mode in ["train", "val", "test"]:
    data_path = BASE_DIR
    data_type_crop = "cropped"
    # train_images = prepare_dataset(data_path, "train", data_type_crop, how="logo")
    test_images = prepare_dataset(data_path, "val", data_type_crop, how="logo")

    torch.save(
        {"test": test_images},
        os.path.join(data_path, f"{data_type_crop}_data_dicts.pth"),
    )
