import json
import math
import os
import sys
from glob import glob
from pathlib import Path
from turtle import width

images_dir = os.path.dirname(os.path.realpath(__file__))
combine_dir = str(Path(images_dir, "combine"))
json_glob = "img*.json"

# img_paths = []
# id_offset = 0
# # for p in glob(f"{images_dir}/many_cards/{img_pattern}"):
# for json_path in glob(f"{images_dir}/single_cards/{json_glob}"):
#     print(os.path.splitext(os.path.basename(json_path))[0] + ".jpg")
#     filename = os.path.basename(json_path)
#     id: int = int(filename[3:-5])  # len("img") == 3, len(".json") == 5
#     # save highest ID of single_cards to offset many_cards IDs so they don't overlap
#     id_offset = id if id > id_offset else id_offset
#     img_paths.append({"id": id, "filename": filename, "full_path": json_path, "type": "single"})
#     Path(images_dir, "train", "images", filename.replace(".json", ".jpg")).write_bytes(Path(json_path.replace(".json", ".jpg")).read_bytes())
# # for p in glob(f"{images_dir}/single_cards/{img_pattern}"):
# for json_path in glob(f"{images_dir}/many_cards/{json_glob}"):
#     filename = os.path.basename(json_path)
#     id: int = int(filename[3:-5]) + id_offset
#     filename = filename.replace(".json", ".jpg")
#     img_paths.append({"id": id, "filename": filename, "full_path": json_path, "type": "many"})
#     Path(images_dir, "train", "images", filename)
# img_paths.sort(key=lambda x: x["id"])
# print(json.dumps(img_paths, indent=4))

train_labels_json: dict = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "background"}, {"id": 1, "name": "card"}]}
validation_labels_json: dict = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "background"}, {"id": 1, "name": "card"}],
}

with open(f"{combine_dir}/merged.json", "r") as f:
    merged: list = json.load(f)

for entry in merged:
    id = int(entry["imagePath"][3:-4])

    image_labels = {
        "id": id,
        "file_name": entry["imagePath"],
    }
    if id in [12, 25, 3, 7]:  # validation set
        validation_labels_json["images"].append(image_labels)
    else:
        train_labels_json["images"].append(image_labels)

    for shape in entry["shapes"]:
        x1 = float(shape["points"][0][0] if shape["points"][0][0] > 0 else 0.0)
        y1 = float(shape["points"][0][1] if shape["points"][0][1] > 0 else 0.0)
        x2 = float(shape["points"][1][0] if shape["points"][1][0] > 0 else 0.0)
        y2 = float(shape["points"][1][1] if shape["points"][1][1] > 0 else 0.0)
        width = float(x2 - x1)
        height = float(y2 - y1)
        if width < 0.1 or height < 0.1:
            print(f"width or height too small: {width}x{height}")
            raise Exception(f"width or height too small: {width}x{height}")

        annotation_labels = {
            "image_id": id,
            "category_id": 1,
            "bbox": [
                round(float(x1), 1),
                round(float(y1), 1),
                round(float(width), 1),
                round(float(height), 1),
            ],
        }
        if id in [12, 25, 3, 7]:  # validation set
            validation_labels_json["annotations"].append(annotation_labels)
        else:
            train_labels_json["annotations"].append(annotation_labels)

# print(json.dumps(train_labels_json, indent=4))
# print(json.dumps(validation_labels_json, indent=4))

# for path_details in img_paths:
#     with open(path_details["full_path"], "r") as f:
#         j = json.loads(f.read())
#     labels_json["images"].append({"id": path_details["id"], "file_name": path_details["filename"], "category_id": 1})
#     for shape in j["shapes"]:
#         labels_json["annotations"].append({"image_id": path_details["id"], "category_id": 1, "bbox": [shape["points"][0][0], shape["points"][0][1], shape["points"][1][0], shape["points"][1][1]]})


# for testing purposes
with open(f"{combine_dir}/train_labels.json", "w") as f:
    json.dump(train_labels_json, f)
with open(f"{combine_dir}/validation_labels.json", "w") as f:
    json.dump(validation_labels_json, f)

# TODO:
# with open(f"{images_dir}/train/labels.json", "w") as f:
#     json.dump(labels_json, f)
# with open(f"{images_dir}/validation/labels.json", "w") as f:
#     json.dump(labels_json, f)
