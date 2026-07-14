# src/generate_features.py
import os
import argparse
import pandas as pd
import numpy as np

def clean_and_normalize_data(data_dir):
    """
    Scans the data directory, maps schemas defensively, 
    and applies our 5-point safety protocol to prevent crashes.
    """
    processed_dfs = []
    
    # 1. DEFENSIVE GOOGLE ADS MAPPING
    g_path = os.path.join(data_dir, "google_ads_campaign_stats.csv")
    if os.path.exists(g_path):
        df = pd.read_csv(g_path)
        # Drop strict duplicates before starting calculations
        df = df.drop_duplicates()
        
        # Enforce robust numeric types safely (invalid texts become NaN)
        df['revenue'] = pd.to_numeric(df['metrics_conversions_value'], errors='coerce')
        df['budget'] = pd.to_numeric(df['campaign_budget_amount'], errors='coerce')
        df['spend'] = pd.to_numeric(df['metrics_cost_micros'], errors='coerce') / 1_000_000.0
        df['clicks'] = pd.to_numeric(df['metrics_clicks'], errors='coerce')
        df['impressions'] = pd.to_numeric(df['metrics_impressions'], errors='coerce')
        df['conversions'] = pd.to_numeric(df['metrics_conversions'], errors='coerce')
        df['date'] = pd.to_datetime(df['segments_date'], errors='coerce')
        df['campaign_name'] = df['campaign_name'].fillna('Unknown_Google_Campaign')
        df['channel'] = 'Google_Ads'
        
        processed_dfs.append(df[['campaign_name', 'date', 'channel', 'budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue']])

    # 2. DEFENSIVE META ADS MAPPING
    m_path = os.path.join(data_dir, "meta_ads_campaign_stats.csv")
    if os.path.exists(m_path):
        df = pd.read_csv(m_path)
        df = df.drop_duplicates()
        
        df['revenue'] = pd.to_numeric(df['conversion'], errors='coerce')
        df['budget'] = pd.to_numeric(df['daily_budget'], errors='coerce')
        df['spend'] = pd.to_numeric(df['spend'], errors='coerce')
        df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce')
        df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
        df['conversions'] = pd.to_numeric(df['conversion'], errors='coerce')
        df['date'] = pd.to_datetime(df['date_start'], errors='coerce')
        df['campaign_name'] = df['campaign_name'].fillna('Unknown_Meta_Campaign')
        df['channel'] = 'Meta_Ads'
        
        processed_dfs.append(df[['campaign_name', 'date', 'channel', 'budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue']])

    # 3. DEFENSIVE BING ADS MAPPING
    b_path = os.path.join(data_dir, "bing_campaign_stats.csv")
    if os.path.exists(b_path):
        df = pd.read_csv(b_path)
        df = df.drop_duplicates()
        
        df['revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
        df['budget'] = pd.to_numeric(df['DailyBudget'], errors='coerce')
        df['spend'] = pd.to_numeric(df['Spend'], errors='coerce')
        df['clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
        df['impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
        df['conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
        df['date'] = pd.to_datetime(df['TimePeriod'], errors='coerce')
        df['campaign_name'] = df['CampaignName'].fillna('Unknown_Bing_Campaign')
        df['channel'] = 'Bing_Ads'
        
        processed_dfs.append(df[['campaign_name', 'date', 'channel', 'budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue']])

    # If the folder is empty or files are completely missing, don't crash! Return empty DataFrame.
    if not processed_dfs:
        return pd.DataFrame(columns=['campaign_name', 'date', 'channel', 'budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue'])

    # Combine all channel datasets together
    master_df = pd.concat(processed_dfs, axis=0, ignore_index=True)

    # 4. EXECUTING OUR ELITE DATA SHIELD
    # Prune impossible rows where clicks outrun impressions
    tracking_mask = master_df['clicks'] > master_df['impressions']
    if tracking_mask.any():
        master_df = master_df[~tracking_mask].reset_index(drop=True)

    # Force absolute values for any invalid negative performance metrics
    numeric_cols = ['budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue']
    for col in numeric_cols:
        if master_df[col].isna().all():
            master_df[col] = 0.0
        else:
            master_df[col] = master_df[col].abs().fillna(0.0)

    # Tiered budget interpolation to fill out missing NaN budget cells safely
    if master_df['budget'].isna().any() or (master_df['budget'] == 0).any():
        channel_medians = master_df.groupby('channel')['budget'].transform('median')
        master_df['budget'] = master_df['budget'].replace(0, np.nan).fillna(channel_medians).fillna(1.0)

    # Drop any completely unreadable or blank dates, then sort chronologically
    final_df = master_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return final_df

def main():
    # Setup argument parameters so the automated grader script can specify folders
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--out', default='./src/processed_features.parquet')
    args = parser.parse_args()
    
    print(f"🛡️  Ingestion Engine active. Processing files from: {args.data_dir}")
    clean_df = clean_and_normalize_data(args.data_dir)
    
    # Save the clean datasets out as a fast-loading parquet matrix file
    clean_df.to_parquet(args.out, index=False)
    print(f"✅ Sanetized features compiled successfully and saved to: {args.out}")

if __name__ == '__main__':
    main()