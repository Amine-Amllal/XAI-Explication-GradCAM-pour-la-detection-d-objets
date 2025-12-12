# Mini-Projet XAI — Explication pour la Détection d'Objets

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📝 Description

Ce projet illustre comment adapter les techniques d'**Explicabilité de l'IA (XAI)**, notamment **Grad-CAM**, pour expliquer les prédictions de modèles de détection d'objets. Contrairement à la classification simple où l'on explique une classe, nous expliquons ici *pourquoi une boîte englobante a été prédite à cet endroit précis*.

---

## 👥 Auteurs

- **Zouga Mouhcine**
- **Amllal Amine**

**Date de réalisation :** 11 décembre 2024

---

## 🎯 Objectifs Pédagogiques

Ce mini-projet s'inscrit dans le cadre d'un cours sur l'Explicabilité de l'IA (XAI) et vise à :

1. **Découvrir** une adaptation de Grad-CAM pour la détection d'objets
2. **Comprendre conceptuellement** comment expliquer les décisions d'un détecteur d'objets
3. **Implémenter** une solution fonctionnelle avec du code production-ready
4. **Analyser** les forces et limites de cette approche XAI

---

## 🔍 Contexte & Motivation

### Le Problème

Les modèles de **détection d'objets** (YOLO, Faster R-CNN, SSD, etc.) sont aujourd'hui omniprésents dans :
- 🚗 Véhicules autonomes
- 📹 Surveillance vidéo
- 🏥 Imagerie médicale
- 🏭 Contrôle qualité industriel

Ces modèles prédisent simultanément :
- **Où** se trouvent les objets (boîtes englobantes)
- **Quoi** sont ces objets (classe)
- **À quel point** le modèle est confiant

Cependant, ces réseaux de neurones profonds sont des **boîtes noires** : ils fournissent des prédictions sans expliquer *pourquoi*.

### Pourquoi l'Explicabilité ?

Dans des contextes critiques (médecine, sécurité, justice), il est essentiel de pouvoir :
- ✅ **Comprendre** les décisions du modèle
- ✅ **Vérifier** qu'il se base sur les bonnes caractéristiques visuelles
- ✅ **Détecter** les biais ou les raccourcis appris

---

## 🧠 Méthode XAI Utilisée : Grad-CAM Adapté

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**Grad-CAM** est une technique qui utilise les gradients (rétropropagation) pour comprendre quelles parties de l'image ont le plus influencé la décision du modèle.

### Adaptation pour la Détection d'Objets

Au lieu de cibler le score d'une classe (classification), nous ciblons :
- Le **score de confiance d'une boîte spécifique**
- Le **score de classe associé à cette boîte**

**Résultat :** Des **heatmaps** (cartes de chaleur) montrant les régions importantes pour chaque détection.

### Famille XAI

- 🏷️ **Type** : Basée sur les gradients
- 🎯 **Explication** : Locale (une boîte spécifique)
- 📊 **Sortie** : Heatmap visuelle
- ⚡ **Performance** : Rapide (un seul forward + backward pass)

---

## 🛠️ Architecture Technique

### Modèle de Détection

- **Architecture** : Faster R-CNN
- **Backbone** : ResNet-50 + FPN (Feature Pyramid Network)
- **Dataset d'entraînement** : COCO (80 classes d'objets)
- **Source** : Modèle pré-entraîné de torchvision

### Pipeline Grad-CAM

```
Image → CNN Backbone → Feature Maps (A^k) → Détection Head → Boîtes + Scores
                              ↓                                    ↓
                        Gradients (∂y/∂A) ←←←←←←←←←←←←←← Score cible (y^c)
                              ↓
                    Poids α = moyenne(gradients)
                              ↓
                    Heatmap = ReLU(Σ α_k · A^k)
```

---

## 📦 Installation & Dépendances

### Prérequis

- Python 3.8+
- CUDA (optionnel, pour GPU)

### Installation

```bash
# Installation des dépendances principales
pip install torch torchvision
pip install grad-cam
pip install opencv-python
pip install requests pillow matplotlib numpy
```

### Packages Utilisés

| Package | Version | Usage |
|---------|---------|-------|
| PyTorch | 2.0+ | Modèle de détection et calcul des gradients |
| Torchvision | 0.15+ | Modèle Faster R-CNN pré-entraîné |
| OpenCV | 4.0+ | Traitement d'images |
| Matplotlib | 3.5+ | Visualisations |
| NumPy | 1.20+ | Calculs numériques |

---

## 🚀 Utilisation

### Exécution du Notebook

1. Ouvrir `Mini_Projet_XAI_Detection_Objets.ipynb`
2. Exécuter les cellules séquentiellement
3. Les images de test sont chargées automatiquement depuis Unsplash

### Sections du Notebook

1. **Installation des dépendances**
2. **Chargement du modèle Faster R-CNN**
3. **Chargement des images de test**
4. **Détection d'objets**
5. **Implémentation de Grad-CAM pour la détection**
6. **Génération des heatmaps d'explication**
7. **Visualisations multiples** (superpositions, comparaisons)
8. **Analyse détaillée** avec métriques quantitatives
9. **Interprétation des résultats**

---

## 📊 Résultats & Visualisations

Le notebook génère plusieurs types de visualisations :

### 1. Heatmaps Grad-CAM
- Heatmap brute (colormap hot/jet)
- Superposition sur l'image originale
- Comparaison de plusieurs colormaps (JET, Inferno)

### 2. Analyses Quantitatives
- **Score d'alignement** : mesure si la heatmap est bien concentrée dans la boîte
- **Statistiques** : valeur max, moyenne, pourcentage de pixels activés
- **Indicateurs de qualité** : 🟢 Excellent / 🟡 Bon / 🔴 Problématique

### 3. Comparaisons Multi-Images
- Traitement automatique de plusieurs images
- Explications pour les 3 meilleures détections par image

---

## 💡 Points Clés à Retenir

### ✅ Forces de Grad-CAM pour la Détection

| Aspect | Évaluation |
|--------|------------|
| **Rapidité** | ⭐⭐⭐⭐⭐ Un seul forward + backward pass |
| **Simplicité** | ⭐⭐⭐⭐ Facile à implémenter et comprendre |
| **Interprétabilité** | ⭐⭐⭐⭐ Heatmaps visuellement intuitives |
| **Flexibilité** | ⭐⭐⭐⭐ Applicable à tout CNN avec feature maps |

**Avantages par rapport à d'autres méthodes :**
- **vs LIME** : Pas besoin de perturbations multiples (plus rapide)
- **vs SHAP** : Pas de calcul combinatoire coûteux
- **vs Saliency Maps** : Plus lisses et moins bruitées

### ⚠️ Limites et Pièges

1. **Résolution limitée** : Les feature maps de la dernière couche sont de basse résolution
2. **Isolation imparfaite** : Difficile d'isoler parfaitement une seule boîte
3. **Sensibilité à l'architecture** : Le choix de la couche cible influence les résultats
4. **Pas d'incertitude** : La heatmap ne quantifie pas l'incertitude
5. **Risque de biais** : Le modèle peut utiliser des corrélations spurieuses

### 🎯 Contextes d'Utilisation

| Contexte | Recommandation |
|----------|----------------|
| **Debugging de modèle** | ✅ Très utile |
| **Communication aux non-experts** | ✅ Les heatmaps sont intuitives |
| **Décisions critiques** | ⚠️ À utiliser en complément |
| **Certification/Audit** | ❌ Insuffisant seul |

---

## 🔬 Extensions Possibles

- **D-RISE** : Méthode de perturbation spécifique à la détection
- **Grad-CAM++** : Version améliorée avec meilleure localisation
- **Score-CAM** : Alternative sans gradients, plus stable
- **Attention Maps** : Pour architectures avec mécanismes d'attention (DETR, ViT)
- **Contrefactuels visuels** : "Que changer pour que la détection disparaisse ?"

---

## 📚 Références

### Articles Scientifiques

1. **Grad-CAM (original)**  
   Selvaraju et al. (2017) - *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"*  
   ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

2. **D-RISE pour la détection**  
   Petsiuk et al. (2021) - *"Black-box Explanation of Object Detectors via Saliency Maps"*  
   CVPR 2021. [arXiv:2006.03204](https://arxiv.org/abs/2006.03204)

3. **Grad-CAM++**  
   Chattopadhay et al. (2018) - *"Grad-CAM++: Generalized Gradient-based Visual Explanations"*  
   WACV 2018.

### Documentation Technique

- [PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)
- [Torchvision Detection Models](https://pytorch.org/vision/stable/models.html#object-detection)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Captum (PyTorch Interpretability)](https://captum.ai/)

---

## 📂 Structure du Projet

```
XAI/
│
├── Mini_Projet_XAI_Detection_Objets.ipynb    # Notebook principal
├── README.md                                  # Ce fichier
└── (images générées lors de l'exécution)
```

---

## 🤝 Contribution

Ce projet est réalisé dans un cadre pédagogique. Les contributions sont les bienvenues pour :
- Tester d'autres modèles de détection (YOLO, DETR)
- Implémenter d'autres méthodes XAI (D-RISE, Score-CAM)
- Améliorer les visualisations
- Ajouter des métriques d'évaluation

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- Équipe PyTorch pour les modèles pré-entraînés
- Jacob Gildenblat pour la librairie pytorch-grad-cam
- Unsplash pour les images de test gratuites
- Professeurs et encadrants du cours XAI

---

## 📧 Contact

Pour toute question ou suggestion :
- **Zouga Mouhcine**
- **Amllal Amine**

---

**Note :** Ce projet démontre que les techniques XAI ne se limitent pas à la classification simple, mais peuvent être adaptées à des tâches complexes comme la détection d'objets, ouvrant la voie à une IA plus transparente et compréhensible.
