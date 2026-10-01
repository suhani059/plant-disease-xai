from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd


# ==========================================
# Plant Disease XAI
# Phase 1 - Dataset Analysis
# ==========================================

DATA_DIR = Path("data/raw/PlantVillage/color")


# ------------------------------------------
# 1. Check dataset path
# ------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_DIR.resolve()}\n"
        "Please check the dataset location."
    )

print("=" * 60)
print("PLANT DISEASE DATASET ANALYSIS")
print("=" * 60)

print(f"\nDataset location:")
print(DATA_DIR.resolve())


# ------------------------------------------
# 2. Find class folders
# ------------------------------------------

class_dirs = sorted(
    [folder for folder in DATA_DIR.iterdir() if folder.is_dir()]
)

print(f"\nNumber of class folders: {len(class_dirs)}")

print("\nClasses:")
for i, class_dir in enumerate(class_dirs, start=1):
    print(f"{i:02d}. {class_dir.name}")


# ------------------------------------------
# 3. Count images per class
# ------------------------------------------

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    
}

class_counts = {}

for class_dir in class_dirs:

    images = [
        file
        for file in class_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    class_counts[class_dir.name] = len(images)


# ------------------------------------------
# 4. Display class distribution
# ------------------------------------------

distribution = pd.DataFrame(
    list(class_counts.items()),
    columns=["Class", "Image_Count"]
)

distribution = distribution.sort_values(
    by="Image_Count",
    ascending=False
)

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print(distribution.to_string(index=False))


# ------------------------------------------
# 5. Dataset statistics
# ------------------------------------------

total_images = distribution["Image_Count"].sum()

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total classes : {len(class_dirs)}")
print(f"Total images  : {total_images}")
print(
    f"Average images/class : "
    f"{distribution['Image_Count'].mean():.2f}"
)
print(
    f"Minimum images/class : "
    f"{distribution['Image_Count'].min()}"
)
print(
    f"Maximum images/class : "
    f"{distribution['Image_Count'].max()}"
)


# ------------------------------------------
# 6. Find sample images
# ------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE IMAGE INFORMATION")
print("=" * 60)

sample_records = []

for class_dir in class_dirs:

    images = [
        file
        for file in class_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    if not images:
        continue

    sample_image = images[0]

    try:
        with Image.open(sample_image) as img:

            sample_records.append({
                "Class": class_dir.name,
                "Filename": sample_image.name,
                "Width": img.width,
                "Height": img.height,
                "Mode": img.mode,
                "Format": img.format
            })

    except Exception as e:

        print(
            f"Could not read {sample_image}: {e}"
        )


sample_df = pd.DataFrame(sample_records)

print(sample_df.to_string(index=False))


# ------------------------------------------
# 7. Check image dimensions
# ------------------------------------------

print("\n" + "=" * 60)
print("CHECKING IMAGE DIMENSIONS")
print("=" * 60)

dimension_counter = Counter()

checked_images = 0
corrupted_images = []

for class_dir in class_dirs:

    images = [
        file
        for file in class_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    for image_path in images:

        try:

            with Image.open(image_path) as img:

                dimension_counter[
                    (img.width, img.height)
                ] += 1

                checked_images += 1

        except Exception:

            corrupted_images.append(str(image_path))


print(f"Images successfully checked: {checked_images}")

print("\nMost common image dimensions:")

for dimension, count in dimension_counter.most_common(10):

    print(
        f"{dimension[0]} x {dimension[1]} : {count} images"
    )


# ------------------------------------------
# 8. Corrupted image check
# ------------------------------------------

print("\n" + "=" * 60)
print("CORRUPTED IMAGE CHECK")
print("=" * 60)

print(
    f"Corrupted/unreadable images: "
    f"{len(corrupted_images)}"
)

if corrupted_images:

    print("\nExamples:")

    for image in corrupted_images[:10]:

        print(image)


# ------------------------------------------
# 9. Save class distribution
# ------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

distribution_path = OUTPUT_DIR / "class_distribution.csv"

distribution.to_csv(
    distribution_path,
    index=False
)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(
    f"\nClass distribution saved to:\n"
    f"{distribution_path.resolve()}"
)