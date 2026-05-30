Markdown
# 🍏 Fruit Object Detection using YOLOv8

Ushbu loyiha kompyuter ko'rishi (Computer Vision) texnologiyalari yordamida rasmlardagi mevalarni aniqlash va ularning sonini hisoblash uchun ishlab chiqilgan. Loyihada **YOLOv8 nano** modeli maxsus ma'lumotlar to'plami ustida qayta o'qitilgan (fine-tuned).

---

## 🛠 Texnologiyalar

* **Framework:** `Ultralytics YOLOv8`
* **Til:** `Python 3.12`
* **Kutubxonalar:** `OpenCV`, `Matplotlib`, `PyTorch`
* **Muhit:** `Jupyter Notebook` & `Kaggle`

---

## 📂 Loyiha Tuzilishi

Repozitoriy quyidagi ierarxiya asosida tashkil etilgan:

```text
├── first-model/
│   └── best.pt                 # 1-o'qitishdan olingan model og'irligi
├── second-model/
│   └── best.pt                 # 2-o'qitishdan olingan model og'irligi
├── results/
│   ├── results.png             # O'qitish grafiklari (Loss / mAP)
│   └── confusion_matrix.png    # Model aniqligi matritsasi
├── train-r/
│   ├── train_batch0.jpg        # Augmentatsiya qilingan namunalar
│   └── val_batch0_pred.jpg     # Validatsiya natijalari rasmi
├── Counting Fruits.ipynb       # Loyihaning asosiy kodi
├── dataset_config.yaml         # Dataset konfiguratsiyasi
├── args.yaml                   # Giperparametrlar sozlamasi
└── README.md
📊 Model Natijalari (Evaluation)
Model o'qitilish jarayoni va aniqlik ko'rsatkichlari:

1. O'qitish va Validatsiya Grafiklari
Loss funksiyalarining kamayishi va mAP50 ko'rsatkichlarining o'sishi:

2. Confusion Matrix
Modelning klasslarni aniqlash bo'yicha aniqlik matritsasi:

🚀 Ishga Tushirish (Quick Start)
1. Kutubxonalarni o'rnatish:
Bash
pip install ultralytics opencv-python matplotlib torch
2. Bitta rasmda modelni sinab ko'rish:
Yaratilgan model yordamida ixtiyoriy rasmdagi mevalarni aniqlash kodi:

Python
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Model va test qilinadigan rasm yo'li
my_model = YOLO("first-model/best.pt")  # yoki second-model/best.pt
image_path = "path/to/your/image.jpg"

# Bashorat qilish
results = my_model(image_path, verbose=False)
result = results[0]

print(f"🤖 Rasmdan jami {len(result.boxes)} ta meva topildi!")

# Natijani vizualizatsiya qilish
annotated_img = result.plot(labels=False, conf=False)
annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(annotated_img)
plt.axis('off')
plt.show()
🎯 Kelajakdagi Rejalar
[ ] Dataset hajmini kattalashtirish orqali aniqlikni oshirish.

[ ] Modelni veb-interfeys (Streamlit yoki FastAPI) orqali integratsiya qilish.
