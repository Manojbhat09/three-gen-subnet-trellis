# Real-ESRGAN
PyTorch implementation of a Real-ESRGAN model trained on custom dataset. This model shows better results on faces compared to the original version. It is also easier to integrate this model into your projects.

> This is not an official implementation. We partially use code from the [original repository](https://github.com/xinntao/Real-ESRGAN)

Real-ESRGAN is an upgraded [ESRGAN](https://arxiv.org/abs/1809.00219) trained with pure synthetic data is capable of enhancing details while removing annoying artifacts for common real-world images. 

You can try it in [google colab](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YlWt--P9w25JUs8bHBOuf8GcMkx-hocP?usp=sharing)

- [Paper (Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data)](https://arxiv.org/abs/2107.10833)
- [Original implementation](https://github.com/xinntao/Real-ESRGAN)
- [Huggingface 🤗](https://huggingface.co/sberbank-ai/Real-ESRGAN)

### Installation

```bash
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

### Usage

---

Basic usage:

```python
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

path_to_image = 'inputs/lr_image.png'
image = Image.open(path_to_image).convert('RGB')

sr_image = model.predict(image)

sr_image.save('results/sr_image.png')
```

### Examples

---

Low quality image:

![](inputs/lr_image.png)

Real-ESRGAN result:

![](results/sr_image.png)

---

Low quality image:

![](inputs/lr_face.png)

Real-ESRGAN result:

![](results/sr_face.png)

## 🚀 Quick Start with Test Scripts

We've included two convenient test scripts to get you started quickly:

### Simple Test Script (`simple_test.py`)

The easiest way to test Real-ESRGAN with command-line arguments:

```bash
# Basic usage (uses defaults)
python simple_test.py

# Custom input/output paths
python simple_test.py --input_path /path/to/your/image.jpg --output_path /path/to/save/result.png

# Short form
python simple_test.py -i /path/to/your/image.jpg -o /path/to/save/result.png

# Different scaling factor (2x, 4x, or 8x)
python simple_test.py --scale 2 --input_path image.jpg --output_path result_2x.png

# Custom processing parameters
python simple_test.py --batch_size 8 --patch_size 256 --input_path image.jpg --output_path result.png
```

### Comprehensive Test Script (`test_realesrgan.py`)

Full-featured script with detailed progress reporting and error handling:

```bash
# Basic usage
python test_realesrgan.py

# Custom parameters
python test_realesrgan.py --input_path my_image.png --output_path results/my_result.png --scale 4
```

### Available Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--input_path` | `-i` | `front_view.png` | Path to input image |
| `--output_path` | `-o` | `results/upscaled.png` | Path to save output |
| `--scale` | `-s` | `4` | Upscaling factor (2, 4, or 8) |
| `--batch_size` | `-b` | `4` | Batch size for processing |
| `--patch_size` | `-p` | `192` | Patch size for processing |

### Get Help

```bash
python simple_test.py --help
python test_realesrgan.py --help
```

### Common Use Cases

**Upscale a specific image:**
```bash
python simple_test.py -i /home/mbhat/my_image.jpg -o results/upscaled.jpg
```

**2x upscaling for faster processing:**
```bash
python simple_test.py -s 2 -i image.jpg -o result_2x.jpg
```

**High-quality processing with larger patches:**
```bash
python simple_test.py -p 512 -b 2 -i image.jpg -o result_hq.png
```

**Process multiple images in sequence:**
```bash
python simple_test.py -i image1.jpg -o results/upscaled1.png
python simple_test.py -i image2.jpg -o results/upscaled2.png
python simple_test.py -i image3.jpg -o results/upscaled3.png
```

## 📁 File Structure

```
Real-ESRGAN/
├── RealESRGAN/           # Main package
│   ├── __init__.py
│   ├── model.py          # RealESRGAN model implementation
│   ├── rrdbnet_arch.py   # Network architecture
│   └── utils.py          # Utility functions
├── weights/              # Model weights (auto-downloaded)
├── results/              # Output directory (auto-created)
├── simple_test.py        # Quick test script
├── test_realesrgan.py    # Comprehensive test script
├── requirements.txt      # Dependencies with exact versions
└── setup.py             # Installation script
```

## 🔧 Installation

### From Source (Recommended)

```bash
git clone <repository-url>
cd Real-ESRGAN
pip install -e .
```

### Dependencies

The package automatically installs compatible versions of:
- `torch==2.5.0+cu121`
- `numpy==2.2.6`
- `opencv-python==4.12.0.88`
- `Pillow==10.4.0`
- `tqdm==4.67.1`
- `huggingface-hub==0.34.0`

## ⚠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `--batch_size` or `--patch_size`
2. **Import errors**: Ensure you're in the correct directory and have installed dependencies
3. **Model download fails**: Check internet connection and HuggingFace access

### Performance Tips

- **Faster processing**: Use smaller `--patch_size` (128 or 192)
- **Higher quality**: Use larger `--patch_size` (256 or 512)
- **Memory optimization**: Reduce `--batch_size` if you encounter CUDA errors

---

Low quality image:

![](inputs/lr_lion.png)

Real-ESRGAN result:

![](results/sr_lion.png)
