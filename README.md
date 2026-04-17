# CNN Model Comparison on Cloud

This project trains multiple CNN architectures on an image dataset, compares their performance, applies transfer learning where supported, and displays the results in a cloud-friendly Streamlit dashboard.

## Supported Models

- LeNet
- ZFNet-style CNN
- AlexNet
- GoogLeNet
- VGG16
- ResNet50
- Inception v3
- MobileNet v3 Large
- SqueezeNet 1.1

## Dataset Format

Use a folder structure like this:

```text
data/
  your_dataset/
    class_1/
      image1.jpg
      image2.jpg
    class_2/
      image3.jpg
      image4.jpg
```

The code automatically splits the dataset into training, validation, and test subsets.

It also supports datasets that already have this structure:

```text
dataset/
  Train/
    class_1/
    class_2/
  Test/
    class_1/
    class_2/
```

`Training/` and `Testing/` directory names are also supported for datasets such as Fruits-360.

## Quick Start

1. Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train all configured models:

```powershell
python train.py --data-dir data/your_dataset --epochs 10 --batch-size 32
```

For a large dataset like Fruits-360, start with a smaller subset:

```powershell
python train.py --data-dir data/fruits-360_100x100/fruits-360 --epochs 3 --batch-size 32 --max-classes 12
```

3. Launch the dashboard locally:

```powershell
streamlit run app.py
```

## Cloud Deployment

The easiest path is Streamlit Community Cloud.

1. Push this folder to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Deploy the repo with `app.py` as the entrypoint.
4. Upload the generated `artifacts/summary.csv` and related output folders if you train elsewhere.

If you want cloud training as well, run the training script in:

- Google Colab
- Kaggle Notebooks
- Paperspace
- AWS EC2 or SageMaker
- Azure ML
- GCP Vertex AI

## Notes

- Transfer learning is used automatically for torchvision backbones when `--mode transfer` or `--mode both` is selected.
- Inception v3 expects larger images; the pipeline handles that through model-specific transforms.
- ZFNet is implemented as a compact approximation because torchvision does not provide an official ZFNet model.
- `max_classes` is useful for large datasets when you want a faster comparison run first.
