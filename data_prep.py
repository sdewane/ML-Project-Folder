import pandas as pd
import numpy as np

def normalize_metro(name):
    if pd.isna(name) or name == "": return "Other"
    main_city = str(name).split('-')[0]
    if ',' in name:
        state = name.split(',')[-1].strip().split('-')[0]
        return f"{main_city}, {state}"
    return main_city

def clean_and_merge_data():
    print("Loading and cleaning primary data (ZHVI & ZORI)...")
    home_values_df = pd.read_csv('data/zhvi.csv', dtype={'RegionName': str})
    rent_prices_df = pd.read_csv('data/zori.csv', dtype={'RegionName': str})

    full_id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName', 'State', 'City', 'Metro', 'CountyName']
    
    home_long = pd.melt(home_values_df, id_vars=full_id_vars, var_name='Date', value_name='Typical_Home_Value')
    home_long['Date'] = pd.to_datetime(home_long['Date'])
    home_long = home_long[home_long['Date'] >= '2015-01-01']
    rent_long = pd.melt(rent_prices_df, id_vars=full_id_vars, var_name='Date', value_name='Typical_Monthly_Rent')
    rent_long['Date'] = pd.to_datetime(rent_long['Date'])
    merged_df = pd.merge(home_long, rent_long[['RegionName', 'Date', 'Typical_Monthly_Rent']], on=['RegionName', 'Date'], how='inner')
    merged_df['Price_to_Rent_Ratio'] = merged_df['Typical_Home_Value'] / (merged_df['Typical_Monthly_Rent'] * 12)
    merged_df['Market_Category'] = np.where(merged_df['Price_to_Rent_Ratio'] < 15, 1, 0)

    print("Merging Days to Pending (Metro Data)...")
    pending_df = pd.read_csv('data/days_pending.csv')
    pending_id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    pending_long = pd.melt(pending_df, id_vars=pending_id_vars, var_name='Date', value_name='Days_To_Pending')
    pending_long['Date'] = pd.to_datetime(pending_long['Date'])
    pending_long = pending_long.rename(columns={'RegionName': 'Metro'})
    merged_df['Metro_Clean'] = merged_df['Metro'].apply(normalize_metro)
    pending_long['Metro_Clean'] = pending_long['Metro'].apply(normalize_metro)

    merged_df = pd.merge(merged_df, pending_long[['Metro_Clean', 'Date', 'Days_To_Pending']], on=['Metro_Clean', 'Date'], how='left')

 
    print("Merging ZHVF (Forecast Data)...")
    forecast_df = pd.read_csv('data/zhvf.csv', dtype={'RegionName': str})
    last_forecast_col = forecast_df.columns[-1]
    forecast_subset = forecast_df[['RegionName', last_forecast_col]].rename(columns={last_forecast_col: 'ZHVF_Forecast'})
    merged_df = pd.merge(merged_df, forecast_subset, on='RegionName', how='left')

    print("Merging Mortgage Payment (Affordability Data)...")
    mort_df = pd.read_csv('data/mortgage_payment.csv', dtype={'RegionName': str})
    mort_id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    
    mort_long = pd.melt(mort_df, id_vars=mort_id_vars, var_name='Date', value_name='Mortgage_Payment')
    mort_long['Date'] = pd.to_datetime(mort_long['Date'])
    merged_df = pd.merge(merged_df, mort_long[['RegionName', 'Date', 'Mortgage_Payment']], on=['RegionName', 'Date'], how='left')

    print("Fetching CPI data from FRED...")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    cpi_df = pd.read_csv(url)
    cpi_df.columns = ['Date', 'CPI']
    cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
    
    merged_df['YearMonth'] = merged_df['Date'].dt.to_period('M')
    cpi_df['YearMonth'] = cpi_df['Date'].dt.to_period('M')
    final_df = pd.merge(merged_df, cpi_df[['YearMonth', 'CPI']], on='YearMonth', how='left')
    final_df['Year'] = final_df['Date'].dt.year
    final_df['Month'] = final_df['Date'].dt.month
    
    print("Performing final data safety sweep...")
    cols_to_fill = ['Days_To_Pending', 'ZHVF_Forecast', 'Mortgage_Payment', 'CPI']
    for col in cols_to_fill:
        if col in final_df.columns:
            median_val = final_df[col].median()
            final_df[col] = final_df[col].fillna(median_val if not pd.isna(median_val) else 0)

    final_df = final_df.drop(columns=['YearMonth', 'Metro_Clean'], errors='ignore')
    final_df = final_df.dropna() 

    print("Engineering 'Momentum' features...")
    final_df['Rent_Affordability_Ratio'] = final_df['Typical_Monthly_Rent'] / (final_df['Mortgage_Payment'] + 1)
    final_df['Market_Heat_Index'] = final_df['ZHVF_Forecast'] / (final_df['Days_To_Pending'] + 1)
    
    print("Engineering advanced time-series and seasonal features...")
    final_df = final_df.sort_values(by=['RegionName', 'Date'])
    final_df['Rent_3mo_Avg'] = final_df.groupby('RegionName')['Typical_Monthly_Rent'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    final_df['Pending_3mo_Avg'] = final_df.groupby('RegionName')['Days_To_Pending'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    final_df['Rent_6mo_Growth'] = final_df.groupby('RegionName')['Typical_Monthly_Rent'].diff(periods=6)
    final_df['Pending_Velocity_3mo'] = final_df.groupby('RegionName')['Days_To_Pending'].diff(periods=3)
    final_df['Is_Peak_Season'] = final_df['Month'].apply(lambda x: 1 if 4 <= x <= 8 else 0)
    final_df = final_df.fillna(0)
    final_df.to_csv('data/final_cleaned_market_data.csv', index=False)
    print(f"Success! Cleaned data saved with {len(final_df)} rows.")
    return final_df
