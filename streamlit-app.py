import streamlit as st
import numpy as np
import pandas as pd
import os
import random
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- Config ---
st.set_page_config(page_title="Pneumonia Detection", layout="wide")
st.title("📷 Détection de Pneumonie sur Radiographies Thoraciques")

# --- Load models ---


@st.cache_resource
def load_models():
    models = {
        "CNN personnalisé": load_model("models/model_CNN_custom.h5"),
        "MobileNetV2": load_model("models/model_MobileNetV2.h5"),
        "ResNet50": load_model("models/model_ResNet50.h5")
    }
    return models


models = load_models()

# --- Helper function ---


def preprocess_image(img: Image.Image):
    img = img.resize((224, 224)).convert("RGB")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# --- Sidebar ---
st.sidebar.header("Exploration des données")
if st.sidebar.checkbox("Afficher un aperçu du dataset"):
    class_counts = {"NORMAL": 1583, "PNEUMONIA": 4273}
    st.sidebar.bar_chart(pd.DataFrame.from_dict(
        class_counts, orient='index', columns=['Nombre']))
    st.sidebar.write("Exemple de radiographies :")

    # Chemins des dossiers
    normal_dir = "data/train/NORMAL/"
    pneumonia_dir = "data/train/PNEUMONIA/"

    # Liste des fichiers images
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

    pneumonia_images = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)
                        if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

    # Sélection aléatoire
    selected_normal = random.choice(normal_images) if normal_images else None
    selected_pneumonia = random.choice(
        pneumonia_images) if pneumonia_images else None

    # Affichage
    col1, col2 = st.sidebar.columns(2)
    if selected_normal:
        col1.image(selected_normal, caption="Normal", use_column_width=True)
    if selected_pneumonia:
        col2.image(selected_pneumonia, caption="Pneumonia",
                   use_column_width=True)


# --- Upload & Predict ---
st.subheader("🔎 Téléversez une image pour prédiction")
file = st.file_uploader("Choisir une image JPG ou PNG",
                        type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file)
    st.image(image, caption="Image chargée", use_column_width=False, width=300)
    img_array = preprocess_image(image)

    st.divider()
    st.subheader("🧠 Résultats des modèles")
    cols = st.columns(3)
    results = []

    for i, (name, model) in enumerate(models.items()):
        pred = model.predict(img_array, verbose=0)
        if pred.shape[1] == 1:
            prob_pneu = float(pred[0][0])
        else:
            prob_pneu = float(pred[0][1])
        prob_norm = 1 - prob_pneu
        label = "PNEUMONIA" if prob_pneu > 0.5 else "NORMAL"
        confidence = round(max(prob_pneu, prob_norm) * 100, 2)
        results.append({"Model": name, "Label": label,
                       "Confidence": confidence})

        with cols[i]:
            st.metric(label=name, value=label, delta=f"{confidence}%")
            st.progress(prob_pneu if label == "PNEUMONIA" else prob_norm)

    # --- Résumé ---
    st.divider()
    st.subheader("🏆 Comparaison des modèles")
    perf_data = pd.DataFrame({
        "Model": [r["Model"] for r in results],
        "Confidence": [r["Confidence"] for r in results]
    }).sort_values(by="Confidence", ascending=False)
    best = perf_data.iloc[0]
    st.success(
        f"Meilleur modèle : {best['Model']} avec {best['Confidence']}% de confiance")

    chart = sns.barplot(data=perf_data, x="Confidence",
                        y="Model", palette="viridis")
    plt.xlabel("Confiance de prédiction (%)")
    plt.title("Comparaison des prédictions")
    st.pyplot(chart.figure)

else:
    st.info("Veuillez téléverser une radiographie pulmonaire pour lancer la prédiction.")
