import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def train_models(data_path='data/final_cleaned_market_data.csv'):
    print(f"Loading pre-cleaned data from {data_path}...")
    df = pd.read_csv(data_path)

    features = [
        'Typical_Monthly_Rent', 'CPI', 'Year', 'Month', 
        'Days_To_Pending', 'ZHVF_Forecast', 'Mortgage_Payment',
        'Rent_Affordability_Ratio', 'Market_Heat_Index',
        'Rent_3mo_Avg', 'Pending_3mo_Avg', 'Rent_6mo_Growth', 
        'Pending_Velocity_3mo', 'Is_Peak_Season'
    ]
   
    X = df[features]
    y = df['Market_Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')

    print("Applying SMOTE to balance the classes...")
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_resampled, y_resampled, train_size=20000, stratify=y_resampled, random_state=42
    )

    results = {'features': features, 'y_test': y_test, 'models': {}}
    #define models
    gb_base = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm_base = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    mlp_base = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    full_stack = StackingClassifier(
        estimators=[('svm', svm_base), ('rf', rf_base), ('mlp', mlp_base), ('gb', gb_base)],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )

    lean_stack = StackingClassifier(
        estimators=[('rf', rf_base), ('gb', gb_base)],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )

    models_to_train = {
        'SVM': svm_base,
        'Random Forest': rf_base,
        'Neural Network': mlp_base,
        'Gradient Boosting': gb_base,
        'Full Stack': full_stack,
        'Lean Stack': lean_stack
    }
    for name, model in models_to_train.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_sample, y_train_sample)
        
        y_probs = model.predict_proba(X_test_scaled)[:, 1]
        threshold = 0.4 
        y_pred = (y_probs >= threshold).astype(int)
        
        print(f"--- {name} Evaluation (Threshold: {threshold}) ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))


        importance = None
        if name == 'SVM' and hasattr(model, 'coef_'):
            importance = model.coef_[0]
        elif name in ['Random Forest', 'Gradient Boosting']: # Added GB here!
            importance = model.feature_importances_

        results['models'][name] = {
            'importance': importance,
            'cm': confusion_matrix(y_test, y_pred),
            'probs': y_probs
        }
        
        joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

    return results