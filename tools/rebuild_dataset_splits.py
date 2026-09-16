#!/usr/bin/env python3
"""Rebuild deterministic, source-aware YOLO train/val/test lists."""

from collections import Counter, defaultdict
from pathlib import Path
import random


ROOT = Path("data/anytrek_multiclass")
SEED = 20260913
RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLITS = ("train", "val", "test")
CALIBRATE_COUNT = 512


def read_all_paths():
    paths = set()
    for image_dir in (ROOT / "images").glob("*"):
        if image_dir.is_dir():
            paths.update(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
    return sorted(paths)


def source_for(path):
    name = path.name.lower()
    if name.startswith("web_"):
        return "web"
    if name.startswith("logo_"):
        return "logo"
    if name[:8].isdigit():
        return "real"
    raise ValueError(f"Unknown image source: {path}")


def label_path(image_path):
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def category_for(image_path):
    labels = label_path(image_path)
    classes = {
        int(fields[0])
        for line in labels.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(fields := line.split()) >= 5
    }
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


def vector(records):
    categories = ("person", "forklift", "mixed", "logo", "empty")
    return [sum(source == "real" and category == key for source, category, _ in records)
            for key in categories]


def score(counts, targets):
    # Relative squared error keeps small but important classes visible.
    result = 0.0
    for actual, target in zip(counts, targets):
        scale = max(target, 1.0)
        result += ((actual - target) / scale) ** 2
    return result


def assign_real_groups(real_records):
    """Assign complete date/hour capture groups to reduce adjacent-frame leakage."""
    by_date = defaultdict(list)
    for record in real_records:
        # Timestamped camera frames in the same hour are kept together.
        by_date[record[2].name[:11]].append(record)

    all_counts = vector(real_records)
    targets = {
        split: [value * RATIOS[split] for value in all_counts]
        for split in ("val", "test")
    }
    groups = sorted(
        by_date.values(),
        key=lambda group: (-len(group), group[0][2].name[:8]),
    )

    best = None
    rng = random.Random(SEED)
    for _ in range(3000):
        order = groups[:]
        rng.shuffle(order)
        order.sort(key=len, reverse=True)
        assigned = {split: [] for split in SPLITS}
        counts = {split: [0] * 5 for split in ("val", "test")}

        for group in order:
            group_counts = vector(group)
            options = []
            for split in ("val", "test", "train"):
                candidate = {
                    key: values[:]
                    for key, values in counts.items()
                }
                if split in counts:
                    candidate[split] = [
                        a + b for a, b in zip(candidate[split], group_counts)
                    ]
                current_score = sum(
                    score(candidate[key], targets[key])
                    for key in ("val", "test")
                )
                # Keep split sizes close to their nominal ratios.
                size_penalty = 0.00001 * abs(
                    len(assigned[split]) + len(group)
                    - len(real_records) * RATIOS[split]
                )
                options.append((current_score + size_penalty, split))
            _, selected = min(options)
            assigned[selected].extend(group)
            if selected in counts:
                counts[selected] = [
                    a + b for a, b in zip(counts[selected], group_counts)
                ]

        final_score = sum(score(counts[key], targets[key]) for key in ("val", "test"))
        final_score += 0.00001 * sum(
            abs(len(assigned[key]) - len(real_records) * RATIOS[key])
            for key in SPLITS
        )
        if best is None or final_score < best[0]:
            best = (final_score, assigned)

    return best[1]


def stratified_individual(records):
    grouped = defaultdict(list)
    for source, category, path in records:
        grouped[(source, category)].append((source, category, path))
    result = {split: [] for split in SPLITS}
    rng = random.Random(SEED)
    for key, group in sorted(grouped.items()):
        rng.shuffle(group)
        n = len(group)
        n_val = round(n * RATIOS["val"])
        n_test = round(n * RATIOS["test"])
        result["val"].extend(group[:n_val])
        result["test"].extend(group[n_val:n_val + n_test])
        result["train"].extend(group[n_val + n_test:])
    return result


def write_lists(assignments):
    for split in SPLITS:
        paths = sorted(str(path) for _, _, path in assignments[split])
        (ROOT / f"{split}.txt").write_text(
            "\n".join(paths) + "\n", encoding="utf-8"
        )


def write_calibration(assignments):
    grouped = defaultdict(list)
    for source, category, path in assignments["train"]:
        grouped[(source, category)].append(path)

    requested = {
        ("real", "person"): 300,
        ("real", "forklift"): 90,
        ("real", "mixed"): 30,
        ("real", "empty"): 40,
        ("web", "forklift"): 26,
        ("web", "empty"): 0,
        ("logo", "logo"): 20,
        ("logo", "empty"): 6,
    }
    rng = random.Random(SEED)
    selected = []
    for key, count in requested.items():
        candidates = grouped[key][:]
        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(f"Not enough calibration candidates for {key}: {len(candidates)}")
        selected.extend(candidates[:count])
    rng.shuffle(selected)
    if len(selected) != CALIBRATE_COUNT:
        raise AssertionError(f"Expected {CALIBRATE_COUNT} calibration images, got {len(selected)}")
    Path("calibrate.txt").write_text(
        "\n".join(str(path) for path in selected) + "\n",
        encoding="utf-8",
    )


def print_summary(assignments):
    for split in SPLITS:
        images = assignments[split]
        source_counts = Counter(source for source, _, _ in images)
        category_counts = Counter(category for _, category, _ in images)
        box_counts = Counter()
        for source, _, path in images:
            for line in label_path(path).read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                fields = line.split()
                if len(fields) >= 5:
                    box_counts[(source, int(fields[0]))] += 1
        print(
            split,
            "images=", len(images),
            "sources=", dict(source_counts),
            "categories=", dict(category_counts),
            "boxes=", dict(box_counts),
        )


def main():
    paths = read_all_paths()
    records = []
    for path in paths:
        if not path.exists() or not label_path(path).exists():
            raise FileNotFoundError(f"Missing image or label: {path}")
        source = source_for(path)
        if source == "real":
            # 20260527 was removed intentionally and must never return.
            if path.name.startswith("20260527_"):
                continue
        records.append((source, category_for(path), path))

    # Stratify by source and image-level category so each split represents
    # the deployment mixture. A future session-id field can replace this
    # with grouped splitting when capture-session metadata is available.
    assignments = stratified_individual(records)
    write_lists(assignments)
    write_calibration(assignments)
    print_summary(assignments)


if __name__ == "__main__":
    main()
