import pandas as pd
import joblib
import os

def find_top_picks(data_path='data/final_cleaned_market_data.csv'):

    if not os.path.exists('lean_stack_model.pkl') or not os.path.exists('scaler.pkl'):
        print("Error: Models not found. Run main.py first!")
        return

    model = joblib.load('lean_stack_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv(data_path)

    features = [
        'Typical_Monthly_Rent', 'CPI', 'Year', 'Month', 
        'Days_To_Pending', 'ZHVF_Forecast', 'Mortgage_Payment',
        'Rent_Affordability_Ratio', 'Market_Heat_Index',
        'Rent_3mo_Avg', 'Pending_3mo_Avg', 'Rent_6mo_Growth', 
        'Pending_Velocity_3mo', 'Is_Peak_Season'
    ]

    latest_date = df['Date'].max()
    current_market = df[df['Date'] == latest_date].copy()
    X_scaled = scaler.transform(current_market[features])
    current_market['Buy_Confidence'] = model.predict_proba(X_scaled)[:, 1]
    top_10 = current_market.sort_values(by='Buy_Confidence', ascending=False).head(10)

    print(f"\n--- TOP 10 REAL ESTATE INVESTMENT PICKS (As of {latest_date}) ---")
    print("-" * 85)
    print(f"{'Zip Code':<10} {'City':<15} {'State':<10} {'Market Heat':<15} {'Buy Confidence':<15}")
    print("-" * 85)
    
    for _, row in top_10.iterrows():
        print(f"{row['RegionName']:<10} {row['City']:<15} {row['State']:<10} "
              f"{row['Market_Heat_Index']:<15.2f} {row['Buy_Confidence']*100:<14.1f}%")

if __name__ == "__main__":
    find_top_picks()