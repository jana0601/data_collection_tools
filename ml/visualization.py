"""
Visualization components for gesture classification results.
Creates comprehensive plots and charts for model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GestureVisualizer:
    """Creates visualizations for gesture classification results."""
    
    def __init__(self, results_dir: str = "ml/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Dict[str, Any]]):
        """Plot comparison of all models."""
        models = list(evaluation_results.keys())
        accuracies = [results['accuracy'] for results in evaluation_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of accuracies
        bars = ax1.bar(models, accuracies, color=sns.color_palette("husl", len(models)))
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Rotate x-axis labels
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart of model performance
        colors = sns.color_palette("husl", len(models))
        wedges, texts, autotexts = ax2.pie(accuracies, labels=models, autopct='%1.1f%%',
                                          colors=colors, startangle=90)
        ax2.set_title('Model Performance Distribution')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str], 
                             model_name: str):
        """Plot confusion matrix for a specific model."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(f'Confusion Matrix - {model_name.title()}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'confusion_matrix_{model_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_reports(self, evaluation_results: Dict[str, Dict[str, Any]], 
                                  class_names: List[str]):
        """Plot detailed classification reports for all models."""
        n_models = len(evaluation_results)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(evaluation_results.items()):
            report = results['classification_report']
            
            # Extract metrics for each class
            metrics = ['precision', 'recall', 'f1-score']
            data = []
            
            for class_name in class_names:
                if class_name in report:
                    row = [report[class_name][metric] for metric in metrics]
                    data.append(row)
                else:
                    data.append([0, 0, 0])
            
            # Create DataFrame
            df = pd.DataFrame(data, index=class_names, columns=metrics)
            
            # Plot heatmap
            sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       ax=axes[i], cbar_kws={'label': 'Score'})
            
            axes[i].set_title(f'Classification Report - {model_name.title()}')
            axes[i].set_xlabel('Metrics')
            axes[i].set_ylabel('Classes')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'classification_reports.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: List[str], top_n: int = 20):
        """Plot feature importance for tree-based models."""
        # Get top N features
        top_indices = np.argsort(feature_importance)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_importance, 
                      color=sns.color_palette("viridis", len(top_features)))
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{top_importance[i]:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_summary(self, training_results: Dict[str, Any]):
        """Plot training summary statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model accuracy comparison
        training_scores = training_results['training_scores']
        models = list(training_scores.keys())
        scores = list(training_scores.values())
        
        bars = axes[0, 0].bar(models, scores, color=sns.color_palette("husl", len(models)))
        axes[0, 0].set_title('Cross-Validation Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        for bar, score in zip(bars, scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 2. Class distribution
        class_names = training_results['class_names']
        n_classes = len(class_names)
        
        axes[0, 1].pie([1] * n_classes, labels=class_names, autopct='%1.1f%%',
                      colors=sns.color_palette("husl", n_classes))
        axes[0, 1].set_title('Class Distribution')
        
        # 3. Data shape information
        data_shape = training_results['data_shape']
        categories = ['Train Samples', 'Test Samples', 'Features']
        values = [data_shape['train_samples'], data_shape['test_samples'], data_shape['features']]
        
        bars = axes[1, 0].bar(categories, values, color=sns.color_palette("Set2", 3))
        axes[1, 0].set_title('Dataset Statistics')
        axes[1, 0].set_ylabel('Count')
        
        for bar, value in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value}', ha='center', va='bottom')
        
        # 4. Model performance comparison (train vs test)
        evaluation_results = training_results['evaluation_results']
        train_scores = [training_scores[model] for model in models]
        test_scores = [evaluation_results[model]['accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, train_scores, width, label='Train (CV)', alpha=0.8)
        axes[1, 1].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        axes[1, 1].set_title('Train vs Test Performance')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_confidence(self, predictions: Dict[str, Any], 
                                 class_names: List[str]):
        """Plot prediction confidence distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract confidence scores
        confidences = []
        predicted_classes = []
        
        for pred in predictions.values():
            if isinstance(pred, dict) and 'confidence' in pred:
                confidences.append(pred['confidence'])
                predicted_classes.append(pred['predicted_gesture'])
        
        if not confidences:
            print("No confidence data available for plotting")
            return
        
        # Create histogram
        ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Prediction Confidence Distribution')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(confidences):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'prediction_confidence.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_performance(self, evaluation_results: Dict[str, Dict[str, Any]], 
                              class_names: List[str]):
        """Plot performance metrics for each class across models."""
        metrics = ['precision', 'recall', 'f1-score']
        n_metrics = len(metrics)
        n_models = len(evaluation_results)
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            # Prepare data
            data = []
            for model_name, results in evaluation_results.items():
                report = results['classification_report']
                model_scores = []
                
                for class_name in class_names:
                    if class_name in report and metric in report[class_name]:
                        model_scores.append(report[class_name][metric])
                    else:
                        model_scores.append(0)
                
                data.append(model_scores)
            
            # Create DataFrame
            df = pd.DataFrame(data, index=evaluation_results.keys(), columns=class_names)
            
            # Plot heatmap
            sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                       ax=axes[i], cbar_kws={'label': metric.title()})
            
            axes[i].set_title(f'{metric.title()} by Class and Model')
            axes[i].set_xlabel('Classes')
            axes[i].set_ylabel('Models')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'class_performance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, training_results: Dict[str, Any]):
        """Create a comprehensive summary report."""
        report_path = self.results_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("GESTURE CLASSIFICATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset information
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 20 + "\n")
            data_shape = training_results['data_shape']
            f.write(f"Training samples: {data_shape['train_samples']}\n")
            f.write(f"Test samples: {data_shape['test_samples']}\n")
            f.write(f"Features: {data_shape['features']}\n")
            f.write(f"Classes: {', '.join(training_results['class_names'])}\n\n")
            
            # Model performance
            f.write("MODEL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            training_scores = training_results['training_scores']
            evaluation_results = training_results['evaluation_results']
            
            for model_name in training_scores.keys():
                train_score = training_scores[model_name]
                test_score = evaluation_results[model_name]['accuracy']
                f.write(f"{model_name}:\n")
                f.write(f"  Training (CV): {train_score:.4f}\n")
                f.write(f"  Test: {test_score:.4f}\n\n")
            
            # Best model
            best_model = max(training_scores.keys(), key=lambda k: training_scores[k])
            f.write(f"BEST MODEL: {best_model}\n")
            f.write(f"Best accuracy: {training_scores[best_model]:.4f}\n")
        
        print(f"Summary report saved to {report_path}")
