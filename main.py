import os
import joblib
from data_prep import clean_and_merge_data
from train_model import train_models
from analytics import generate_plots
from top_picks import find_top_picks 

def main():
    print("=== Real Estate Market ML Project Pipeline ===")
    
    data_path = 'data/final_cleaned_market_data.csv'
    results_path = 'training_results.pkl'

    if not os.path.exists(data_path):
        clean_and_merge_data()
    else:
        print(f"Skipping data prep. Found existing file at: {data_path}")


    if os.path.exists(results_path):
        print(f"\n--- Loading saved training results from {results_path} ---")
        results = joblib.load(results_path)
    else:
        print("\n--- No saved results found. Starting training... ---")
        results = train_models(data_path=data_path)
        joblib.dump(results, results_path)
        print(f"Training results cached at {results_path}")

    generate_plots(results, data_path=data_path)
    
    find_top_picks(data_path=data_path)

    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()