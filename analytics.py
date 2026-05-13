import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc

sns.set_theme(style="whitegrid")

def generate_plots(results, data_path='data/final_cleaned_market_data.csv'):
    print("Generating Visualizations for all models...")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    features = results['features']
    y_test = results['y_test']

    output_folder = "visualizations"
    os.makedirs(output_folder, exist_ok=True)

    model_names = list(results['models'].keys())
    num_models = len(model_names)
    
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    fig.suptitle('Model Comparison: Prediction Errors (Confusion Matrices)', fontsize=18, fontweight='bold')

    if num_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        cm = results['models'][model_name]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Rent (0)', 'Buy (1)'], 
                    yticklabels=['Rent (0)', 'Buy (1)'], ax=axes[i], cbar=False)
        axes[i].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Actual Market Reality', fontsize=10)
        axes[i].set_xlabel('Model Prediction', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, '2_Model_Comparison_CM.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    
    colors = {
        'SVM': '#e74c3c', 
        'Random Forest': '#3498db', 
        'Neural Network': '#9b59b6', 
        'Gradient Boosting': '#f39c12', 
        'Full Stack': '#2ecc71',
        'Lean Stack': '#1abc9c'
    }

    for model_name, model_data in results['models'].items():
        fpr, tpr, _ = roc_curve(y_test, model_data['probs'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=3, color=colors.get(model_name, 'black'),
                 label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guessing (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Performance Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '1_ROC_Curve_Comparison.png'), dpi=300)
    plt.close()


    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Feature Importance: Why the Models are Picking "Buy" Markets', fontsize=16, fontweight='bold')

    rf_imp = pd.DataFrame({
        'Feature': features, 
        'Importance': results['models']['Random Forest']['importance']
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=rf_imp, color='#3498db', ax=axes[0])
    axes[0].set_title('Random Forest: Feature Impact', fontsize=14)
    axes[0].set_xlabel('Relative Importance')

    gb_imp = pd.DataFrame({
        'Feature': features, 
        'Importance': results['models']['Gradient Boosting']['importance']
    }).sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=gb_imp, color='#f39c12', ax=axes[1])
    axes[1].set_title('Gradient Boosting: Feature Impact', fontsize=14)
    axes[1].set_xlabel('Relative Importance')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, '3_Feature_Importance_Comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 10))
    corr_cols = features + ['Market_Category']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Feature Correlation: How Variables Link to "Buy" Signals', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '4_Feature_Correlation.png'), dpi=300)
    plt.close()
    
    print(f"Success! All charts saved to the '{output_folder}' directory.")