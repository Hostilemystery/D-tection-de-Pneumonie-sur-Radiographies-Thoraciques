# 📊 Application de Détection de Pneumonie sur Radiographies (Streamlit)

Cette application Streamlit permet d'explorer le jeu de données **Chest X-Ray Images (Pneumonia)** et de prédire si une radiographie téléversée indique une pneumonie ou non à l'aide de **trois modèles de deep learning** :

- Un **CNN personnalisé**
- **MobileNetV2** (Transfer Learning)
- **ResNet50** (Transfer Learning)

L'interface est moderne, intuitive et affiche des visualisations des données ainsi que des comparaisons de performance entre les modèles.

---

## ✨ Fonctionnalités

- 📊 Exploration des données : distribution des classes, affichage d'exemples.
- 📸 Téléversement d'une radiographie thoracique.
- 🤖 Prédiction de la classe (Normal ou Pneumonia) par 3 modèles.
- ⚖️ Comparaison des performances (accuracy, AUC).
- 🏆 Mise en avant du meilleur modèle.

---

## ⚠️ Prérequis

### 1. Données

Téléchargez le dataset depuis Kaggle : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Ensuite, **copiez les sous-dossiers suivants** :

```
chest_xray/
├── train/
├── val/
└── test/
```

Dans le dossier `data/` du projet. Structure finale attendue :

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 2. Modèles pré-entraînés

Assurez-vous que les modèles suivants sont bien présents dans le dossier `models/` :

```
models/
├── model_CNN_custom.h5
├── model_MobileNetV2.h5
└── model_ResNet50.h5
```

Ces modèles ont été entraînés sur les données du dataset préalablement.

### 3. Images d'exemple

Ajoutez deux images d'exemple dans le dossier `examples/` pour l'exploration visuelle :

```
examples/
├── NORMAL_example.jpg
└── PNEUMONIA_example.jpg
```

---

## 🚀 Installation

### 1. Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Fichier `requirements.txt` recommandé :

```txt
streamlit
numpy
pandas
matplotlib
seaborn
pillow
tensorflow
scikit-learn
```

---

## 🔄 Lancer l'application

Depuis le dossier contenant le script `streamlit-app.py` :

```bash
streamlit run streamlit-app.py
```

Puis ouvrez [http://localhost:8501](http://localhost:8501) dans votre navigateur.

---

## 🔄 Structure du projet

```
.
├── streamlit-app.py                    # Code principal de l'application
├── models/                             # Modèles sauvegardés en .h5
├── radiographie_classification.ipynb   # Entrainement de l'algo
├── data/                               # Dossier contenant train/val/test
├── requirements.txt                    # Dépendances
├── README.md                           # Ce fichier
```

---

## 🌟 Créateurs

Projet réalisé dans le cadre du TP de Deep Learning à l'Efrei. Pour toute question ou amélioration, contactez Gedeon Freddy NANJI ENGA et Anthony CORMEAUX.

---

🚀 Bon test et exploration des radiographies thoraciques !
