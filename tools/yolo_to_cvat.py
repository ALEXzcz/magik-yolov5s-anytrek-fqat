#!/usr/bin/env python3
"""Convert YOLO prediction files to CVAT XML and pixel-coordinate JSON."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABELS = {0: "person", 1: "forklift"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    return parser.parse_args()


def clamp(value, lower, upper):
    return max(lower, min(value, upper))


def read_boxes(label_file, width, height):
    boxes = []
    seen = set()
    if label_file.exists():
        for line_number, line in enumerate(label_file.read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) not in (5, 6):
                raise ValueError(
                    f"{label_file}:{line_number}: expected 5 or 6 fields, got {len(fields)}"
                )

            class_id = int(float(fields[0]))
            if class_id not in LABELS:
                continue

            center_x, center_y, box_width, box_height = map(float, fields[1:5])
            confidence = float(fields[5]) if len(fields) == 6 else None
            x1 = clamp((center_x - box_width / 2) * width, 0.0, float(width))
            y1 = clamp((center_y - box_height / 2) * height, 0.0, float(height))
            x2 = clamp((center_x + box_width / 2) * width, 0.0, float(width))
            y2 = clamp((center_y + box_height / 2) * height, 0.0, float(height))

            if x2 <= x1 or y2 <= y1:
                continue
            box = {
                "label": LABELS[class_id],
                "class_id": class_id,
                "confidence": confidence,
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
            key = (
                box["class_id"],
                box["x1"],
                box["y1"],
                box["x2"],
                box["y2"],
                box["confidence"],
            )
            if key not in seen:
                seen.add(key)
                boxes.append(box)
    return boxes


def format_number(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_xml(images):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "cvat_prelabels"
    labels = ET.SubElement(task, "labels")
    for label_name in LABELS.values():
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = label_name
        ET.SubElement(label, "type").text = "any"
        ET.SubElement(label, "attributes")

    for image_id, image_data in enumerate(images):
        image_node = ET.SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": image_data["name"],
                "width": str(image_data["width"]),
                "height": str(image_data["height"]),
            },
        )
        for box in image_data["boxes"]:
            ET.SubElement(
                image_node,
                "box",
                {
                    "label": box["label"],
                    "xtl": format_number(box["x1"]),
                    "ytl": format_number(box["y1"]),
                    "xbr": format_number(box["x2"]),
                    "ybr": format_number(box["y2"]),
                    "occluded": "0",
                    "source": "auto",
                    "z_order": "0",
                },
            )
    return ET.ElementTree(root)


def main():
    args = parse_args()
    image_paths = sorted(
        path for path in args.images.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No supported images found in {args.images}")

    image_data = []
    total_boxes = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        label_file = args.labels / f"{image_path.stem}.txt"
        boxes = read_boxes(label_file, width, height)
        image_data.append(
            {
                "name": image_path.name,
                "width": width,
                "height": height,
                "boxes": boxes,
            }
        )
        total_boxes += len(boxes)

    args.xml.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    xml_tree = build_xml(image_data)
    if hasattr(ET, "indent"):
        ET.indent(xml_tree, space="  ")
    xml_tree.write(args.xml, encoding="utf-8", xml_declaration=True)
    args.json.write_text(json.dumps({"images": image_data}, ensure_ascii=False, indent=2) + "\n")

    print(f"Images: {len(image_data)}")
    print(f"Boxes: {total_boxes}")
    print(f"CVAT XML: {args.xml}")
    print(f"Pixel JSON: {args.json}")


if __name__ == "__main__":
    main()
