"""
Downloads RAF-DB from Kaggle and merges with FER-2013 into a unified dataset.

FER-2013 is expected at: /workspace/data/train/ and /workspace/data/test/
RAF-DB is downloaded and merged into: /workspace/data_merged/train/ and /workspace/data_merged/test/

Class folders (unified names):
  angry, disgusted, fearful, happy, sad, surprised, neutral
"""
import os, glob, shutil, zipfile
from pathlib import Path

FER_TRAIN   = "/workspace/data/train"
FER_TEST    = "/workspace/data/test"
RAF_DIR     = "/workspace/raf_db"
OUT_TRAIN   = "/workspace/data_merged/train"
OUT_TEST    = "/workspace/data_merged/test"

CLASSES = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]

# RAF-DB emotion label -> unified class name
# Labels in list_patition_label.txt: 1=Surprise 2=Fear 3=Disgust 4=Happiness 5=Sadness 6=Anger 7=Neutral
RAF_LABEL_MAP = {
    "1": "surprised",
    "2": "fearful",
    "3": "disgusted",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}

# FER-2013 folder name -> unified class name (some datasets use different spellings)
FER_FOLDER_MAP = {
    "angry": "angry", "anger": "angry",
    "disgusted": "disgusted", "disgust": "disgusted",
    "fearful": "fearful", "fear": "fearful",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "surprised": "surprised", "surprise": "surprised",
    "neutral": "neutral",
}


def make_dirs():
    for split in [OUT_TRAIN, OUT_TEST]:
        for cls in CLASSES:
            os.makedirs(f"{split}/{cls}", exist_ok=True)


def copy_fer():
    print("\n[FER-2013] Copying...")
    counts = {cls: 0 for cls in CLASSES}
    for split_src, split_dst in [(FER_TRAIN, OUT_TRAIN), (FER_TEST, OUT_TEST)]:
        for folder in os.listdir(split_src):
            cls = FER_FOLDER_MAP.get(folder.lower())
            if cls is None:
                print(f"  Unknown FER folder: {folder}, skipping")
                continue
            src_dir = f"{split_src}/{folder}"
            dst_dir = f"{split_dst}/{cls}"
            files = glob.glob(f"{src_dir}/*.jpg") + glob.glob(f"{src_dir}/*.png")
            for p in files:
                dst = f"{dst_dir}/fer_{Path(p).name}"
                shutil.copy2(p, dst)
                counts[cls] += 1
    print(f"  Copied: {sum(counts.values())} images  {counts}")


def download_raf():
    print("\n[RAF-DB] Downloading from Kaggle...")
    os.makedirs(RAF_DIR, exist_ok=True)
    ret = os.system(
        f"kaggle datasets download -d shuvoalok/raf-db-dataset -p {RAF_DIR} --unzip"
    )
    if ret != 0:
        print("  kaggle download failed — trying alternative slug...")
        ret = os.system(
            f"kaggle datasets download -d apollo2506/facial-dataset -p {RAF_DIR} --unzip"
        )
    return ret == 0


def find_raf_label_file():
    for pattern in [
        f"{RAF_DIR}/**/list_patition_label.txt",
        f"{RAF_DIR}/**/*label*.txt",
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def copy_raf():
    print("\n[RAF-DB] Parsing and copying...")

    label_file = find_raf_label_file()
    if label_file is None:
        print("  ERROR: Could not find RAF-DB label file. Skipping RAF-DB.")
        print(f"  Files found in {RAF_DIR}:")
        for f in glob.glob(f"{RAF_DIR}/**/*", recursive=True)[:20]:
            print(f"    {f}")
        return

    print(f"  Label file: {label_file}")
    label_dir = os.path.dirname(label_file)

    # Find image directory (aligned or original)
    img_dirs = (
        glob.glob(f"{RAF_DIR}/**/aligned", recursive=True)
        or glob.glob(f"{RAF_DIR}/**/Image/aligned", recursive=True)
        or [label_dir]
    )
    img_dir = img_dirs[0]
    print(f"  Image dir: {img_dir}")

    entries = open(label_file).read().strip().splitlines()
    counts = {cls: 0 for cls in CLASSES}
    skipped = 0

    for line in entries:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        fname, label = parts[0], parts[1]
        cls = RAF_LABEL_MAP.get(label)
        if cls is None:
            skipped += 1
            continue

        # RAF-DB label file uses original names, aligned files have _aligned suffix
        stem = Path(fname).stem
        candidates = (
            glob.glob(f"{img_dir}/{stem}_aligned.jpg")
            + glob.glob(f"{img_dir}/{stem}_aligned.png")
            + glob.glob(f"{img_dir}/{fname}")
        )
        if not candidates:
            skipped += 1
            continue

        src = candidates[0]
        split = "train" if fname.startswith("train") else "test"
        dst_dir = OUT_TRAIN if split == "train" else OUT_TEST
        dst = f"{dst_dir}/{cls}/raf_{Path(src).name}"
        shutil.copy2(src, dst)
        counts[cls] += 1

    print(f"  Copied: {sum(counts.values())} images  {counts}")
    if skipped:
        print(f"  Skipped: {skipped}")


def print_summary():
    print("\n=== Merged dataset summary ===")
    for split, split_dir in [("train", OUT_TRAIN), ("test", OUT_TEST)]:
        total = 0
        row = []
        for cls in CLASSES:
            n = len(glob.glob(f"{split_dir}/{cls}/*.jpg") + glob.glob(f"{split_dir}/{cls}/*.png"))
            row.append(f"{cls}:{n}")
            total += n
        print(f"  {split:5s}  total={total}  {', '.join(row)}")


if __name__ == "__main__":
    make_dirs()
    copy_fer()
    ok = download_raf()
    if ok:
        copy_raf()
    else:
        print("[RAF-DB] Download failed — continuing with FER-2013 only")
    print_summary()
    print("\nDone. Run: python train.py")
