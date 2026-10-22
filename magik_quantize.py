import torch
import argparse
import torch.nn as nn
import random
import numpy as np
import os
import math
import yaml
from pathlib import Path

from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.fqat.quantization.quantization_fqat import MagikQuantizer
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python import quantize_ops
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.calibration_data_reader import seed_all
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.data_reader import Yolov5DataReader
from ingenic_magik_trainingkit.QuantizationTrainingPlugin.python.fqat.quantization.utils import magik_export_onnx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed_all(1000)

DEFAULT_CALIBRATE_PATH = "calibrate.txt"
DEFAULT_DATA_CFG = "data/anytrek_multiclass.yaml"
DEFAULT_CALIBRATE_SAMPLES = 128
CALIBRATE_QUOTAS = {
    64: {
        ("real", "person"): 38,
        ("real", "forklift"): 11,
        ("real", "mixed"): 4,
        ("real", "empty"): 5,
        ("web", "forklift"): 3,
        ("logo", "logo"): 2,
        ("logo", "empty"): 1,
    },
    128: {
        ("real", "person"): 50,
        ("real", "forklift"): 38,
        ("real", "mixed"): 18,
        ("real", "empty"): 12,
        ("web", "forklift"): 0,
        ("logo", "logo"): 7,
        ("logo", "empty"): 3,
    },
}


def _read_data_yaml(data_cfg):
    with open(data_cfg, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_list_path(data_cfg, list_path):
    list_path = Path(list_path)
    if list_path.is_absolute():
        return list_path
    cwd_path = list_path.resolve()
    if cwd_path.exists():
        return cwd_path
    data_dir = Path(data_cfg).resolve().parent
    return (data_dir / list_path).resolve()


def _resolve_train_image_path(train_root, image_path):
    image_path = Path(image_path)
    if image_path.is_absolute():
        return image_path
    cwd_path = image_path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (train_root / image_path).resolve()


def _load_existing_calibrate_entries(calibrate_path):
    path = Path(calibrate_path)
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _entries_all_exist(entries):
    return bool(entries) and all(Path(entry).exists() for entry in entries)


def _calibration_source(path):
    name = Path(path).name.lower()
    if name.startswith("web_"):
        return "web"
    if name.startswith("logo_"):
        return "logo"
    return "real"


def _calibration_category(path):
    label_path = Path(str(path).replace("/images/", "/labels/")).with_suffix(".txt")
    classes = set()
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            classes.add(int(fields[0]))
    classes = frozenset(classes)
    if not classes:
        return "empty"
    if classes == {0}:
        return "person"
    if classes == {1}:
        return "forklift"
    if classes == {0, 1}:
        return "mixed"
    if classes == {2}:
        return "logo"
    return "other"


def rebuild_calibrate_file(calibrate_path=DEFAULT_CALIBRATE_PATH,
                           data_cfg=DEFAULT_DATA_CFG,
                           sample_count=DEFAULT_CALIBRATE_SAMPLES,
                           seed=1000):
    data = _read_data_yaml(data_cfg)
    train_list_path = _resolve_list_path(data_cfg, data["train"])
    train_root = train_list_path.parent.resolve()
    lines = [line.strip() for line in train_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    candidates = []
    for line in lines:
        img_path = _resolve_train_image_path(train_root, line)
        if img_path.exists():
            candidates.append(str(img_path))

    if not candidates:
        raise FileNotFoundError(f"No valid calibration images found from {train_list_path}")

    # Keep calibration representative when an old or missing file triggers
    # automatic regeneration.
    if sample_count in CALIBRATE_QUOTAS:
        requested = CALIBRATE_QUOTAS[sample_count]
        grouped = {}
        for candidate in candidates:
            key = (_calibration_source(candidate), _calibration_category(candidate))
            grouped.setdefault(key, []).append(candidate)
        rng = random.Random(seed)
        selected = []
        for key, count in requested.items():
            pool = grouped.get(key, [])[:]
            if len(pool) < count:
                # Dataset merges may not preserve source filename prefixes.
                # Fall back to the same label category from any source.
                category = key[1]
                pool = [path for (source, cat), paths in grouped.items()
                        if cat == category for path in paths]
            rng.shuffle(pool)
            if len(pool) < count:
                raise ValueError(
                    f"Not enough calibration candidates for {key}: "
                    f"{len(pool)} < {count}"
                )
            selected.extend(pool[:count])
        rng.shuffle(selected)
        sample_size = len(selected)
    else:
        sample_size = min(sample_count, len(candidates))
        rng = random.Random(seed)
        selected = rng.sample(candidates, sample_size)
    Path(calibrate_path).write_text("\n".join(selected) + "\n", encoding="utf-8")
    return selected


def get_magik_quantizer(bit=4, calibrate_path=DEFAULT_CALIBRATE_PATH,
                        data_cfg=DEFAULT_DATA_CFG,
                        calibrate_samples=DEFAULT_CALIBRATE_SAMPLES):
    entries = _load_existing_calibrate_entries(calibrate_path)
    if len(entries) != calibrate_samples or not _entries_all_exist(entries):
        rebuild_calibrate_file(
            calibrate_path=calibrate_path,
            data_cfg=data_cfg,
            sample_count=calibrate_samples,
        )
    calibration_batch_size = 1
    calibration_batches = math.ceil(calibrate_samples / calibration_batch_size)
    cal_data_reader = Yolov5DataReader(
        path=calibrate_path,
        nbatchs=calibration_batches,
        batch_size=calibration_batch_size,
        img_size=(640, 640),
        mean=0,
        var=255
    )  # mean and variance
    # get quantize_parameter
    quantize_parameter = {"weight_bitwidth" : bit,
                          "output_bitwidth" : bit,
                          "target_device" : "T32", # platform
                          "feature_quantize_method" : "DAQS",
                          "feature_calibration_method" : "MSE",
                          "weight_quantize_method" : "DAQS",
                          "weight_calibration_method" : "ADMM",
                          "feature_calibration_loop_size" : 80, # iteration times of feature calibration
                          "weight_calibration_loop_size" : 200, # iteration times of weight calibration
                          "fixpoint" : True, # fixpoint can be aligned with board end data
                          "set_ahead_layers_8bit" : 2, # set the first n layers to 8 bit
                          "set_hinder_layers_8bit" : 2, # set the last n layers to 8 bit
                          "input_mean" : [0, 0, 0], # mean
                          "input_var" : [255, 255, 255], # variance
                          "quantize_model_input" : False, # quantize input of the network or not
                        #   "debug_info" : True
                         }

    magik_quantizer = MagikQuantizer(quantize_parameter,cal_data_reader)

    return magik_quantizer

@torch.no_grad()
def get_and_save_quant_model(magik_quantizer, weight, cfg, onnx):
    from models.yolo import Model
    with open(cfg, encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)
    net = Model(cfg, ch=3, nc=model_cfg.get('nc', 1)).cuda()
    chkpt = torch.load(args.weights)
    net.load_state_dict(chkpt['model'].float().state_dict(), strict=True)
    net.to(device)
    net.eval()

    net = magik_quantizer.quant(net)
    # magik_quantizer.run_debug(net)

    chkpt['ema'] = net
    if onnx:
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        dst_path = "checkpoint/8W8F/weights/best-tmp.onnx"
        magik_export_onnx(net, dummy_input, dst_path)
        print("---convert---over----")
        exit()
    torch.save(chkpt, "checkpoint/8W8F/weights/best-tmp.pt")
    print("pqt model over----!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Quant')
    parser.add_argument("--weights", type=str, default="checkpoint/32W32F/weights/best-tmp.pt", help="path to model")
    parser.add_argument("--cfg", type=str, default="models/yolov5n.yaml", help="path to model")
    parser.add_argument("--calibrate-path", type=str, default=DEFAULT_CALIBRATE_PATH, help="path to calibration image list")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_CFG, help="path to data yaml for rebuilding calibration list")
    parser.add_argument("--calibrate-samples", type=int, default=DEFAULT_CALIBRATE_SAMPLES, help="number of calibration images")
    parser.add_argument('--bit', type=int, default=8, help='You should set bit to 32 or 8 or 4')
    parser.add_argument('--onnx', action='store_true', help='use OpenCV DNN for ONNX inference')
    args = parser.parse_args()
    magik_quantizer = get_magik_quantizer(args.bit, args.calibrate_path, args.data, args.calibrate_samples)
    get_and_save_quant_model(magik_quantizer, args.weights, args.cfg, args.onnx)
