# Satellite Imagery Based Property Evaluation

## Property Valuation: Multimodal Modeling Strategy

This repository contains a multimodal machine learning pipeline that integrates high-resolution satellite imagery with tabular data to predict property market values.

1. Project Setup
Prerequisites
Python 3.8+

Kaggle Account (for T4 GPU access used during training)

Google Maps Static API Key (required for data_fetcher.ipynb)

## Installation

Clone the repository and install the required dependencies:

pip install catboost lightgbm pandas numpy scikit-learn torch torchvision shap matplotlib folium

## File Structure & Execution Order


1. data_fetcher.ipynb, Acquires satellite tiles via Google Maps Static API based on coordinates.
2. embedding_dino_for_satimg.ipynb , Extracts 768-dimensional visual embeddings using a pre-trained DINOv2 (base) model.
3. eda_final.ipynb, "Conducts post-model analysis, including residual plots and SHAP explainability."
4. Dinov2-Attention maps, Contains attention maps of the model
5. feature_engineering_final.ipynb, Performs cyclical month encoding and K-Means geoclustering.
6. model_training_satimg_and_tabular, Trains the final CatBoost Regressor using native embedding support (R2=0.9118) and also contains post modelling analysis.

## Modeling Methodology

### Data Fusion:

The project utilizes Late Fusion by passing raw 768-dimensional DINOv2 embeddings directly into CatBoost's embedding_features parameter. This approach outperformed PCA-reduced baselines by preserving the visual manifold structure.

### Optimization & Regularization:

Log Transformation: The target variable price is optimized on a log1p scale to handle right-skewed distributions.

Regularization: Applied l2_leaf_reg and adjusted tree depth to mitigate overfitting, maintaining a healthy gap between training and validation scores.

Cross-Validation: Results are validated using a robust 5-Fold Cross-Validation strategy.

## Explainability:

Attention Maps: DINOv2 self-attention maps are used to verify the model attends to salient visual features like rooftops and pools.

SHAP Analysis: Provides a quantitative breakdown of feature contributions to individual property valuations.

## Visualizations

property_explorer.html: Interactive tool for viewing property features and attention maps.

wealth_heatmap.html: Geospatial heatmap of predicted market values across the study area.




