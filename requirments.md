# 📋 Requirements for Plant Disease Classification App

This project uses a DenseNet121 model trained on 72+ plant disease classes, served through a Flask web app.

To run this project locally, make sure the following Python packages are installed:

## 🔧 Core Dependencies

| Package     | Minimum Version | Purpose                                 |
|-------------|------------------|-----------------------------------------|
| `torch`     | `2.0.0` or above | Deep learning framework (PyTorch)       |
| `torchvision` | `0.15.0`+      | For image preprocessing & transforms    |
| `flask`     | `2.0.0` or above | Web framework for serving the app       |
| `pillow`    | -                | Image file handling (JPEG, PNG, etc.)   |
| `numpy`     | -                | Numerical operations                    |

## 📁 Optional (Used in Some Variants)

| Package     | Purpose                                      |
|-------------|----------------------------------------------|
| `pandas`    | For CSV/JSON data handling (optional)        |
| `gunicorn`  | For deploying the Flask app (optional)       |

## 📌 Install All at Once

You can install all required packages using the following:

```bash
pip install torch torchvision flask pillow numpy
```

## 📌 Including Optional Packages

You can install all required packages using the following:
```bash
pip install torch torchvision flask pillow numpy pandas gunicorn
```
