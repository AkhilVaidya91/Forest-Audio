#!/usr/bin/env python3
"""
Forest Audio Classification ML-Ops Pipeline
Integrates preprocessing, model training, and MLFlow for complete ML lifecycle management
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.models.signature import infer_signature

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

class ForestAudioMLOpsePipeline:
    def __init__(self, config_path=None):
        """Initialize the ML-Ops pipeline with configuration"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        self._setup_mlflow()
        
    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            "data": {
                "dataset_dir": r"C:\Users\Lenovo\Documents\projects\research\Major-Project\Forest-Audio\data\audio",
                "metadata_file": r"C:\Users\Lenovo\Documents\projects\research\Major-Project\Forest-Audio\data\metadata.csv",
                "output_dir": r"C:\Users\Lenovo\Documents\projects\research\Major-Project\Forest-Audio\data\processed_audio",
                "sample_rate": 44100,
                "fixed_duration": 5.0,
                "n_mfcc": 128,
                "hop_length": 512,
                "n_mels": 128
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
                "cv_folds": 3,
                "scoring_metric": "f1"
            },
            "mlflow": {
                "experiment_name": "forest_audio_classification",
                "tracking_uri": "./mlruns",
                "artifact_location": "./mlflow_artifacts"
            },
            "binary_mapping": {
                0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 0, 9: 1,
                10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 0, 18: 0, 19: 0,
                20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Deep merge configurations
            for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"forest_audio_pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline logging initialized")
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config["data"]["output_dir"],
            os.path.join(self.config["data"]["output_dir"], "mel_images"),
            self.config["mlflow"]["artifact_location"],
            "dvc_data",
            "plots"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info("Directories created successfully")
    
    def _setup_mlflow(self):
        """Setup MLFlow tracking"""
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.config["mlflow"]["experiment_name"])
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.config["mlflow"]["experiment_name"],
                    artifact_location=self.config["mlflow"]["artifact_location"]
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            self.logger.warning(f"MLFlow setup warning: {e}")
            experiment_id = mlflow.create_experiment(
                self.config["mlflow"]["experiment_name"],
                artifact_location=self.config["mlflow"]["artifact_location"]
            )
        
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        self.logger.info(f"MLFlow experiment set: {self.config['mlflow']['experiment_name']}")
    
    def _calculate_data_hash(self, data_path):
        """Calculate hash of data for versioning"""
        hash_md5 = hashlib.md5()
        if os.path.isfile(data_path):
            with open(data_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def load_and_explore_data(self):
        """Load metadata and perform initial EDA"""
        self.logger.info("Loading and exploring data...")
        
        with mlflow.start_run(run_name="data_exploration"):
            # Load metadata
            self.metadata = pd.read_csv(self.config["data"]["metadata_file"])
            
            # Log dataset info
            mlflow.log_param("total_samples", len(self.metadata))
            mlflow.log_param("metadata_file", self.config["data"]["metadata_file"])
            mlflow.log_param("data_hash", self._calculate_data_hash(self.config["data"]["metadata_file"]))
            
            # EDA
            class_counts = self.metadata["Class Name"].value_counts()
            mlflow.log_param("num_classes", len(class_counts))
            
            # Create and save class distribution plot
            plt.figure(figsize=(12, 8))
            sns.countplot(y="Class Name", data=self.metadata, 
                         order=self.metadata["Class Name"].value_counts().index)
            plt.title("Class Distribution")
            plt.tight_layout()
            plot_path = "plots/class_distribution.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            plt.close()
            
            # Log class distribution as artifact
            class_dist_df = pd.DataFrame({
                'class_name': class_counts.index,
                'count': class_counts.values
            })
            class_dist_path = "plots/class_distribution.csv"
            class_dist_df.to_csv(class_dist_path, index=False)
            mlflow.log_artifact(class_dist_path)
            
            self.logger.info(f"Data exploration complete. Found {len(self.metadata)} samples across {len(class_counts)} classes")
            
            return self.metadata
    
    def preprocess_audio(self, file_path, sr=None, fixed_duration=None):
        """Preprocess individual audio file"""
        if sr is None:
            sr = self.config["data"]["sample_rate"]
        if fixed_duration is None:
            fixed_duration = self.config["data"]["fixed_duration"]
            
        try:
            y, sr = librosa.load(file_path, sr=sr)
            
            # Pad or Trim to Fixed Duration
            target_len = int(fixed_duration * sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            else:
                y = y[:target_len]
            
            # Normalize amplitude
            y = librosa.util.normalize(y)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config["data"]["n_mfcc"])
            mfcc = np.mean(mfcc.T, axis=0)  # aggregate
            
            return y, mfcc
        except Exception as e:
            self.logger.error(f"Error preprocessing {file_path}: {e}")
            return None, None
    
    def extract_mel_spectrogram(self, y, sr=None, n_mels=None, hop_length=None, fixed_duration=None):
        """Extract mel spectrogram from audio"""
        if sr is None:
            sr = self.config["data"]["sample_rate"]
        if n_mels is None:
            n_mels = self.config["data"]["n_mels"]
        if hop_length is None:
            hop_length = self.config["data"]["hop_length"]
        if fixed_duration is None:
            fixed_duration = self.config["data"]["fixed_duration"]
            
        try:
            target_len = int(fixed_duration * sr)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            else:
                y = y[:target_len]

            # Mel-Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            return mel_spec_db
        except Exception as e:
            self.logger.error(f"Error creating mel-spectrogram: {e}")
            return None
    
    def preprocess_dataset(self):
        """Preprocess entire dataset and create features"""
        self.logger.info("Starting dataset preprocessing...")
        
        with mlflow.start_run(run_name="data_preprocessing"):
            # Log preprocessing parameters
            mlflow.log_params({
                "sample_rate": self.config["data"]["sample_rate"],
                "fixed_duration": self.config["data"]["fixed_duration"],
                "n_mfcc": self.config["data"]["n_mfcc"],
                "hop_length": self.config["data"]["hop_length"],
                "n_mels": self.config["data"]["n_mels"]
            })
            
            features = []
            labels = []
            failed_files = []
            
            # Create mel spectrogram images directory
            img_output_dir = os.path.join(self.config["data"]["output_dir"], "mel_images")
            os.makedirs(img_output_dir, exist_ok=True)
            
            for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Processing audio files"):
                file_path = os.path.join(self.config["data"]["dataset_dir"], row["Dataset File Name"])
                
                if not os.path.exists(file_path):
                    self.logger.warning(f"File not found: {file_path}")
                    failed_files.append(file_path)
                    continue
                
                y, mfcc = self.preprocess_audio(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(row["Class ID"])
                    
                    # Create and save mel spectrogram
                    mel_spec = self.extract_mel_spectrogram(y)
                    if mel_spec is not None:
                        # Get binary class label
                        original_class = row["Class ID"]
                        binary_class = self.config["binary_mapping"].get(original_class, 0)
                        
                        # Save spectrogram image
                        filename = f"img_{idx}_{original_class}_{binary_class}.png"
                        save_path = os.path.join(img_output_dir, filename)
                        
                        plt.figure(figsize=(4, 4))
                        librosa.display.specshow(mel_spec, sr=self.config["data"]["sample_rate"], 
                                               hop_length=self.config["data"]["hop_length"],
                                               x_axis=None, y_axis=None, cmap="magma")
                        plt.axis("off")
                        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                        plt.close()
                else:
                    failed_files.append(file_path)
            
            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)
            
            # Log preprocessing results
            mlflow.log_param("total_processed_files", len(features))
            mlflow.log_param("failed_files_count", len(failed_files))
            mlflow.log_param("features_shape", f"{features.shape}")
            
            # Apply binary mapping
            y_binary = np.array([self.config["binary_mapping"][label] for label in labels])
            
            # Feature Scaling
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, y_binary, 
                test_size=self.config["training"]["test_size"],
                stratify=y_binary,
                random_state=self.config["training"]["random_state"]
            )
            
            # Log split information
            mlflow.log_params({
                "train_size": X_train.shape[0],
                "test_size": X_test.shape[0],
                "train_positive_ratio": np.mean(y_train),
                "test_positive_ratio": np.mean(y_test)
            })
            
            # Save processed data
            output_dir = self.config["data"]["output_dir"]
            np.save(os.path.join(output_dir, "X_train.npy"), X_train)
            np.save(os.path.join(output_dir, "X_test.npy"), X_test)
            np.save(os.path.join(output_dir, "y_train.npy"), y_train)
            np.save(os.path.join(output_dir, "y_test.npy"), y_test)
            
            # Save scaler
            import joblib
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)
            
            # Log data artifacts
            for file_name in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]:
                mlflow.log_artifact(os.path.join(output_dir, file_name))
            
            self.logger.info(f"Preprocessing complete! Processed {len(features)} files")
            self.logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test, scaler
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"forest_audio_pipeline_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.log_file = log_file
    
    def get_model_configs(self):
        """Get model configurations for grid search"""
        configs = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(class_weight="balanced", 
                                              random_state=self.config["training"]["random_state"]),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            "SVM": {
                "model": SVC(class_weight="balanced", probability=True, 
                           random_state=self.config["training"]["random_state"]),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf"],
                    "gamma": ["auto"]
                }
            }
        }
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            configs["LightGBM"] = {
                "model": LGBMClassifier(class_weight="balanced", 
                                      random_state=self.config["training"]["random_state"],
                                      force_col_wise=True, verbose=-1),
                "params": {
                    "n_estimators": [100, 200],
                    "num_leaves": [31, 63],
                    "learning_rate": [0.01, 0.1, 0.2]
                }
            }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            configs["XGBoost"] = {
                "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                     random_state=self.config["training"]["random_state"],
                                     scale_pos_weight=1),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }
            }
        
        return configs
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models with MLFlow tracking"""
        self.logger.info("Starting model training and evaluation...")
        
        model_configs = self.get_model_configs()
        results = []
        best_model_info = {"f1_score": 0, "model": None, "model_name": "", "run_id": ""}
        
        for model_name, config in model_configs.items():
            self.logger.info(f"Training {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_training", nested=True):
                try:
                    # Log model configuration
                    mlflow.log_params(config["params"])
                    mlflow.log_param("model_type", model_name)
                    
                    # Grid Search
                    grid = GridSearchCV(
                        estimator=config["model"],
                        param_grid=config["params"],
                        scoring=self.config["training"]["scoring_metric"],
                        cv=self.config["training"]["cv_folds"],
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    
                    # Predictions
                    y_pred = best_model.predict(X_test)
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    # Calculate AUC if probabilities available
                    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1_score": f1,
                        "auc": auc if auc else 0
                    })
                    
                    # Log best parameters
                    mlflow.log_params(grid.best_params_)
                    
                    # Create and log confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                               xticklabels=[0, 1], yticklabels=[0, 1])
                    plt.title(f"Confusion Matrix - {model_name}")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    cm_path = f"plots/confusion_matrix_{model_name.replace(' ', '_')}.png"
                    plt.savefig(cm_path)
                    mlflow.log_artifact(cm_path)
                    plt.close()
                    
                    # Log model
                    signature = infer_signature(X_train, best_model.predict(X_train))
                    mlflow.sklearn.log_model(
                        best_model, 
                        f"model_{model_name.replace(' ', '_')}",
                        signature=signature
                    )
                    
                    # Track best model
                    if f1 > best_model_info["f1_score"]:
                        best_model_info.update({
                            "f1_score": f1,
                            "model": best_model,
                            "model_name": model_name,
                            "run_id": mlflow.active_run().info.run_id
                        })
                    
                    # Store results
                    result = {
                        "Model": model_name,
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "AUC": auc if auc else "N/A",
                        "Best Params": grid.best_params_,
                        "Run ID": mlflow.active_run().info.run_id
                    }
                    results.append(result)
                    
                    # Print results
                    self.logger.info(f"Results for {model_name}:")
                    self.logger.info(f"  Accuracy: {acc:.4f}")
                    self.logger.info(f"  Precision: {prec:.4f}")
                    self.logger.info(f"  Recall: {rec:.4f}")
                    self.logger.info(f"  F1 Score: {f1:.4f}")
                    if auc:
                        self.logger.info(f"  AUC: {auc:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
                    continue
        
        # Register best model
        if best_model_info["model"] is not None:
            self._register_best_model(best_model_info)
        
        return results, best_model_info
    
    def _register_best_model(self, best_model_info):
        """Register the best performing model"""
        self.logger.info(f"Registering best model: {best_model_info['model_name']}")
        
        try:
            # Register model
            model_uri = f"runs:/{best_model_info['run_id']}/model_{best_model_info['model_name'].replace(' ', '_')}"
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name="forest_audio_best_classifier"
            )
            
            # Add model description
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name="forest_audio_best_classifier",
                version=registered_model.version,
                description=f"Best performing model: {best_model_info['model_name']} with F1 Score: {best_model_info['f1_score']:.4f}"
            )
            
            self.logger.info(f"Model registered successfully with version {registered_model.version}")
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
    
    def create_comparison_plots(self, results):
        """Create comparison plots for all models"""
        self.logger.info("Creating model comparison plots...")
        
        with mlflow.start_run(run_name="model_comparison"):
            results_df = pd.DataFrame(results)
            
            # Save results CSV
            results_path = "plots/model_comparison_results.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(results_path)
            
            # Performance comparison plot
            plt.figure(figsize=(14, 8))
            metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
            results_df.set_index("Model")[metrics_to_plot].plot(kind="bar")
            plt.title("Model Performance Comparison")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.legend(loc="lower right")
            plt.xticks(rotation=45)
            plt.tight_layout()
            comparison_plot_path = "plots/model_performance_comparison.png"
            plt.savefig(comparison_plot_path)
            mlflow.log_artifact(comparison_plot_path)
            plt.close()
            
            # Best model metrics
            best_model_row = results_df.loc[results_df["F1 Score"].idxmax()]
            mlflow.log_params({
                "best_model": best_model_row["Model"],
                "best_f1_score": best_model_row["F1 Score"],
                "best_accuracy": best_model_row["Accuracy"]
            })
            
            self.logger.info("Model comparison plots created and logged")
            
            return results_df
    
    def save_pipeline_summary(self, results_df, best_model_info):
        """Save comprehensive pipeline summary"""
        summary = {
            "pipeline_run_timestamp": datetime.now().isoformat(),
            "data_config": self.config["data"],
            "training_config": self.config["training"],
            "binary_mapping": self.config["binary_mapping"],
            "model_results": results_df.to_dict("records"),
            "best_model": {
                "name": best_model_info["model_name"],
                "f1_score": best_model_info["f1_score"],
                "run_id": best_model_info["run_id"]
            },
            "log_file": str(self.log_file)
        }
        
        summary_path = "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Log summary to current MLFlow run
        if mlflow.active_run():
            mlflow.log_artifact(summary_path)
        
        self.logger.info(f"Pipeline summary saved to {summary_path}")
        
        return summary
    
    def create_dvc_pipeline(self):
        """Create DVC pipeline configuration"""
        dvc_yaml = {
            "stages": {
                "preprocess": {
                    "cmd": "python forest_audio_mlops_pipeline.py --stage preprocess",
                    "deps": [
                        self.config["data"]["dataset_dir"],
                        self.config["data"]["metadata_file"]
                    ],
                    "outs": [
                        self.config["data"]["output_dir"]
                    ]
                },
                "train": {
                    "cmd": "python forest_audio_mlops_pipeline.py --stage train",
                    "deps": [
                        self.config["data"]["output_dir"]
                    ],
                    "metrics": [
                        "metrics.json"
                    ],
                    "plots": [
                        "plots/"
                    ]
                }
            }
        }
        
        import yaml
        with open("dvc.yaml", "w") as f:
            yaml.dump(dvc_yaml, f, default_flow_style=False)
        
        self.logger.info("DVC pipeline configuration created")
    
    def run_full_pipeline(self):
        """Run the complete ML-Ops pipeline"""
        self.logger.info("Starting Forest Audio Classification ML-Ops Pipeline")
        
        try:
            # Data exploration
            metadata = self.load_and_explore_data()
            
            # Preprocessing
            X_train, X_test, y_train, y_test, scaler = self.preprocess_dataset()
            
            # Model training and evaluation
            results, best_model_info = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            # Create comparison plots
            results_df = self.create_comparison_plots(results)
            
            # Save pipeline summary
            summary = self.save_pipeline_summary(results_df, best_model_info)
            
            # Create DVC pipeline
            self.create_dvc_pipeline()
            
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Best model: {best_model_info['model_name']} with F1 Score: {best_model_info['f1_score']:.4f}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_stage(self, stage):
        """Run specific pipeline stage"""
        if stage == "preprocess":
            metadata = self.load_and_explore_data()
            X_train, X_test, y_train, y_test, scaler = self.preprocess_dataset()
            return "Preprocessing completed"
        
        elif stage == "train":
            # Load preprocessed data
            output_dir = self.config["data"]["output_dir"]
            X_train = np.load(os.path.join(output_dir, "X_train.npy"))
            X_test = np.load(os.path.join(output_dir, "X_test.npy"))
            y_train = np.load(os.path.join(output_dir, "y_train.npy"))
            y_test = np.load(os.path.join(output_dir, "y_test.npy"))
            
            # Train models
            results, best_model_info = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            results_df = self.create_comparison_plots(results)
            summary = self.save_pipeline_summary(results_df, best_model_info)
            
            # Save metrics for DVC
            metrics = {
                "best_f1_score": best_model_info["f1_score"],
                "best_model": best_model_info["model_name"]
            }
            with open("metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            return "Training completed"
        
        else:
            raise ValueError(f"Unknown stage: {stage}")


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Forest Audio Classification ML-Ops Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--stage", type=str, choices=["preprocess", "train", "full"], 
                       default="full", help="Pipeline stage to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ForestAudioMLOpsePipeline(config_path=args.config)
    
    try:
        if args.stage == "full":
            summary = pipeline.run_full_pipeline()
            print("\n" + "="*50)
            print("PIPELINE SUMMARY")
            print("="*50)
            print(f"Best Model: {summary['best_model']['name']}")
            print(f"Best F1 Score: {summary['best_model']['f1_score']:.4f}")
            print(f"Log File: {summary['log_file']}")
            print(f"Total Models Trained: {len(summary['model_results'])}")
        else:
            result = pipeline.run_stage(args.stage)
            print(f"Stage '{args.stage}' completed: {result}")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


# Additional utility functions for manual usage
class ModelInference:
    """Class for model inference and deployment"""
    
    def __init__(self, model_name="forest_audio_best_classifier", model_version="latest"):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        """Load registered model from MLFlow"""
        try:
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"Model {self.model_name} v{self.model_version} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict_audio_file(self, audio_file_path, config_path=None):
        """Predict class for a single audio file"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Load pipeline config to get preprocessing parameters
        pipeline = ForestAudioMLOpsePipeline(config_path=config_path)
        
        # Preprocess audio
        y, mfcc = pipeline.preprocess_audio(audio_file_path)
        if mfcc is None:
            raise ValueError(f"Failed to preprocess audio file: {audio_file_path}")
        
        # Scale features (note: in production, you'd want to save and load the scaler)
        if self.scaler is None:
            print("Warning: Using model without proper feature scaling")
            features = mfcc.reshape(1, -1)
        else:
            features = self.scaler.transform(mfcc.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
        
        return {
            "prediction": int(prediction),
            "probability": probability.tolist() if probability is not None else None
        }


# Configuration template for easy customization
def create_config_template():
    """Create a configuration template file"""
    config_template = {
        "data": {
            "dataset_dir": r"C:\Users\Lenovo\Documents\projects\research\Major-Project\Forest-Audio\data\audio",
            "metadata_file": r"C:\Users\Lenovo\Documents\projects\research\Major-Project\Forest-Audio\data\metadata.csv",
            "output_dir": r"C:\Users\Lenovo\Documents\projects\research\Major-Project\Forest-Audio\data\processed_audio",
            "sample_rate": 44100,
            "fixed_duration": 5.0,
            "n_mfcc": 128,
            "hop_length": 512,
            "n_mels": 128
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "cv_folds": 3,
            "scoring_metric": "f1"
        },
        "mlflow": {
            "experiment_name": "forest_audio_classification",
            "tracking_uri": "./mlruns",
            "artifact_location": "./mlflow_artifacts"
        },
        "binary_mapping": {
            "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 1, "8": 0, "9": 1,
            "10": 1, "11": 1, "12": 1, "13": 1, "14": 1, "15": 1, "16": 1, "17": 0, "18": 0, "19": 0,
            "20": 0, "21": 0, "22": 0, "23": 0, "24": 0, "25": 0, "26": 0, "27": 0
        }
    }
    
    with open("pipeline_config_template.json", "w") as f:
        json.dump(config_template, f, indent=2)
    
    print("Configuration template created: pipeline_config_template.json")
    print("Modify this file with your specific paths and parameters")


if __name__ == "__main__":
    # Check if user wants to create config template
    if len(sys.argv) > 1 and sys.argv[1] == "--create-config":
        create_config_template()
        sys.exit(0)
    
    main()


# =====================================================
# USAGE INSTRUCTIONS:
# =====================================================
"""
SETUP:
1. Install required packages:
   pip install mlflow pandas numpy scikit-learn librosa matplotlib seaborn tqdm joblib
   pip install lightgbm xgboost  # optional but recommended
   pip install dvc  # for data version control

2. Create configuration file (optional):
   python forest_audio_mlops_pipeline.py --create-config
   # Edit pipeline_config_template.json with your paths

RUNNING THE PIPELINE:

1. Full Pipeline:
   python forest_audio_mlops_pipeline.py

2. With custom config:
   python forest_audio_mlops_pipeline.py --config your_config.json

3. Specific stages:
   python forest_audio_mlops_pipeline.py --stage preprocess
   python forest_audio_mlops_pipeline.py --stage train

4. View MLFlow UI:
   mlflow ui
   # Open http://localhost:5000 in browser

DVC INTEGRATION:
1. Initialize DVC:
   dvc init
   
2. Add data to DVC:
   dvc add data/audio
   dvc add data/metadata.csv
   
3. Run DVC pipeline:
   dvc repro

INFERENCE:
# Load and use trained model
inference = ModelInference()
result = inference.predict_audio_file("path/to/new/audio.wav")
print(result)

MLFLOW FEATURES USED:
- Experiment tracking
- Parameter logging
- Metric logging
- Artifact logging (plots, data, models)
- Model registration
- Model versioning
- Run comparison
"""