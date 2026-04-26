from pathlib import Path
from PIL import Image


# =========================
# 1. Config
# =========================
input_dir = Path("qualitative_figures")
output_dir = Path("qualitative_figures_combined")
output_dir.mkdir(exist_ok=True)

case_names = [
    "case_A_medium-difficulty",
    "case_B_challenging",
    "case_C_failed_samples",
]

# 是否统一高度
resize_to_same_height = True

# 两张图之间的留白
gap = 30

# 背景颜色
bg_color = "white"


# =========================
# 2. Helpers
# =========================
def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_same_height(img: Image.Image, target_height: int) -> Image.Image:
    if img.height == target_height:
        return img
    new_width = int(img.width * target_height / img.height)
    return img.resize((new_width, target_height), Image.LANCZOS)


def combine_horizontal(left_img: Image.Image, right_img: Image.Image, gap: int = 30) -> Image.Image:
    if resize_to_same_height:
        target_height = max(left_img.height, right_img.height)
        left_img = resize_same_height(left_img, target_height)
        right_img = resize_same_height(right_img, target_height)

    combined_width = left_img.width + gap + right_img.width
    combined_height = max(left_img.height, right_img.height)

    canvas = Image.new("RGB", (combined_width, combined_height), color=bg_color)
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width + gap, 0))
    return canvas


# =========================
# 3. Combine each case
# =========================
for case_name in case_names:
    waveform_path = input_dir / f"waveform_{case_name}.png"
    spectrogram_path = input_dir / f"spectrogram_{case_name}.png"

    if not waveform_path.exists():
        raise FileNotFoundError(f"Missing waveform image: {waveform_path}")
    if not spectrogram_path.exists():
        raise FileNotFoundError(f"Missing spectrogram image: {spectrogram_path}")

    waveform_img = load_image(waveform_path)
    spectrogram_img = load_image(spectrogram_path)

    # 左：waveform，右：spectrogram
    combined_img = combine_horizontal(waveform_img, spectrogram_img, gap=gap)

    out_png = output_dir / f"combined_{case_name}.png"
    out_pdf = output_dir / f"combined_{case_name}.pdf"

    combined_img.save(out_png)
    combined_img.save(out_pdf, resolution=300.0)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")