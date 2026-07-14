import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import operator
from typing import TypedDict, List, Dict, Any, Annotated
from dotenv import load_dotenv
from google import genai

# Load environment variables (.env)
load_dotenv()

# -----------------------------------------------------------------
# SYSTEM SETUP & USER INTERFACE CONFIGURATION
# -----------------------------------------------------------------
st.set_page_config(
    page_title="NetElixir AIgnition 3.0 | Marketing Forecast Utility", 
    layout="wide"
)

st.title(" 📈 Enterprise Multi-Agent Forecasting & Portfolio Utility")
st.markdown("### Powered by Advanced Bayesian Predictive Engines")
st.write("---")

# -----------------------------------------------------------------
# STATE SCHEMA (TYPEDDICT WITH REDUCERS)
# -----------------------------------------------------------------
class ForeCastingState(TypedDict):
    """State schema governing our pipeline."""
    anamoly_logs: Annotated[List[str], operator.add]
    raw_data_status: str
    cleaned_data: Dict[str, Any]       
    forecast_results: Dict[str, Any]
    simulation_results: Dict[str, Any]
    executive_summary: str

# -----------------------------------------------------------------
# CONTROL PANEL SIDEBAR
# -----------------------------------------------------------------
st.sidebar.header("🕹️ Configuration Control Panel")

planning_horizon = st.sidebar.selectbox(
    "Select Aggregate Planning Horizon",
    options=[30, 60, 90],
    index=0,
    help="Select the aggregate scenario planning horizon required by the brief."
)

future_target_budget = st.sidebar.number_input(
    "Enter Future Portfolio Budget Input ($)",
    min_value=1000.0,
    max_value=5000000.0,
    value=50000.0,
    step=5000.0,
    help="The total budget allocation pool that Workstation 3 optimizes."
)

st.sidebar.subheader("📂 Ingestion Influx Engines")
google_file = st.sidebar.file_uploader("Upload Google Ads Stats CSV (Optional)", type=["csv"])
meta_file = st.sidebar.file_uploader("Upload Meta Ads Stats CSV (Optional)", type=["csv"])
bing_file = st.sidebar.file_uploader("Upload Bing Ads Stats CSV (Optional)", type=["csv"])


# -----------------------------------------------------------------
# PIPELINE WORKSTATION AGENT NODES
# -----------------------------------------------------------------

def workstation_1_ingest(state: ForeCastingState, g_file, m_file, b_file) -> Dict[str, Any]:
    """Node 1: Schema Standardizer & Data Sanitizer Engine with Campaign-Specific Lifespan calculations."""
    processed_dfs = []
    logs = []
    
    if g_file is not None:
        df = pd.read_csv(g_file)
        df['revenue'] = pd.to_numeric(df['metrics_conversions_value'], errors='coerce')
        df['budget'] = pd.to_numeric(df['campaign_budget_amount'], errors='coerce')
        df['spend'] = pd.to_numeric(df['metrics_cost_micros'], errors='coerce') / 1_000_000
        df['clicks'] = pd.to_numeric(df['metrics_clicks'], errors='coerce')
        df['impressions'] = pd.to_numeric(df['metrics_impressions'], errors='coerce')
        df['conversions'] = pd.to_numeric(df['metrics_conversions'], errors='coerce')
        df['date'] = pd.to_datetime(df['segments_date'], errors='coerce')
        df['campaign_name'] = df['campaign_name'].fillna('Unknown_Google_Campaign')
        df['channel'] = 'Google_Ads'
        processed_dfs.append(df[['campaign_name', 'date', 'channel', 'budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue']])
        logs.append("Workstation 1: Successfully normalized Google Ads schema matrices.")

    if m_file is not None:
        df = pd.read_csv(m_file)
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
        logs.append("Workstation 1: Successfully normalized Meta Ads schema matrices.")

    if b_file is not None:
        df = pd.read_csv(b_file)
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
        logs.append("Workstation 1: Successfully normalized Bing Ads schema matrices.")

    if not processed_dfs:
        return {"anamoly_logs": ["Workstation 1 [HALT]: No input files detected."], "raw_data_status": "failed", "cleaned_data": {}}

    master_df = pd.concat(processed_dfs, axis=0, ignore_index=True)

    tracking_mask = master_df['clicks'] > master_df['impressions']
    if tracking_mask.any():
        master_df = master_df[~tracking_mask].reset_index(drop=True)
        logs.append("Workstation 1 [CLEAN]: Eradicated tracking rows showing clicks > impressions.")

    for col in ['budget', 'spend', 'clicks', 'impressions', 'conversions', 'revenue']:
        if (master_df[col] < 0).any():
            master_df[col] = master_df[col].abs()
            logs.append(f"Workstation 1 [FIX]: Converted invalid negative indices in '{col}' to absolute scalars.")

    if master_df['budget'].isna().any():
        channel_medians = master_df.groupby('channel')['budget'].transform('median')
        master_df['budget'] = master_df['budget'].fillna(channel_medians).fillna(master_df['budget'].median())
        logs.append("Workstation 1 [IMPUTE]: Multi-channel tiered budget median interpolation applied.")

    final_df = master_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # Calculate unique active lifespan per campaign
    lifespans = final_df.groupby(['channel', 'campaign_name'])['date'].agg(
        min_date='min',
        max_date='max'
    ).reset_index()
    lifespans['campaign_active_days'] = (lifespans['max_date'] - lifespans['min_date']).dt.days + 1
    
    # Merge lifespan back into the dataframe so downstream nodes have access
    final_df = final_df.merge(lifespans[['channel', 'campaign_name', 'campaign_active_days']], on=['channel', 'campaign_name'], how='left')
    
    return {"anamoly_logs": logs, "raw_data_status": "success", "cleaned_data": {"master_df": final_df}}


def workstation_2_forecaster(state: ForeCastingState, horizon: int) -> Dict[str, Any]:
    """
    Node 2: Probabilistic Aggregate Scenario Forecaster.
    Employs Empirical Bayes Shrinkage to pull volatile NTM campaigns back to portfolio averages.
    """
    master_df = state["cleaned_data"].get("master_df")
    if master_df is None or master_df.empty:
        return {"anamoly_logs": ["Workstation 2 [HALT]: Cleaned historical data stream is missing."], "forecast_results": {}}

    # Calculate global portfolio daily statistics to establish a prior
    global_total_revenue = master_df['revenue'].sum()
    global_active_days = len(master_df['date'].unique())
    global_campaign_count = len(master_df.groupby(['channel', 'campaign_name']))
    
    # Baseline benchmark metrics
    global_daily_revenue_prior = global_total_revenue / (global_campaign_count * global_active_days) if global_campaign_count > 0 else 0.0

    # Include campaign_active_days directly in groupby key to preserve it across aggregation bounds safely
    daily_history = master_df.groupby(['date', 'channel', 'campaign_name', 'campaign_active_days'])[['spend', 'revenue']].sum().reset_index()
    grouped = daily_history.groupby(['channel', 'campaign_name'])
    forecasts = {}
    
    shrinkage_half_life = 14  # Confidence threshold
    logs = [f"Workstation 2: Initializing forecasting over {horizon}-day scope."]

    for (channel_name, campaign_name), group in grouped:
        key = f"{channel_name} | {campaign_name}"
        group = group.sort_values('date')
        
        # Pull precomputed campaign lifespan
        n_c = int(group.iloc[0]['campaign_active_days'])
        
        recent_window = group['revenue'].tail(30)
        if len(recent_window) < 1:  
            continue
            
        raw_daily_rev = recent_window.mean()
        std_daily_rev = recent_window.std() if len(recent_window) > 1 and recent_window.std() > 0 else (raw_daily_rev * 0.1)
        
        # Empirical Bayes Shrinkage
        w_c = n_c / (n_c + shrinkage_half_life)
        smoothed_daily_rev = (w_c * raw_daily_rev) + ((1 - w_c) * global_daily_revenue_prior)
        
        # Apply strict data density penalty for brand new campaigns (under 7 days active)
        if n_c < 7:
            density_penalty = n_c / 7.0
            smoothed_daily_rev *= density_penalty
            
        expected_aggregate_revenue = smoothed_daily_rev * horizon
        std_error_aggregate = std_daily_rev * np.sqrt(horizon) * (1 - w_c + 0.1)
        
        lower_bound = max(0.0, expected_aggregate_revenue - (1.96 * std_error_aggregate))
        upper_bound = expected_aggregate_revenue + (1.96 * std_error_aggregate)
        
        forecasts[key] = {
            "channel": channel_name,
            "campaign": campaign_name,
            "lower_bound_revenue": lower_bound,
            "expected_revenue": expected_aggregate_revenue,
            "upper_bound_revenue": upper_bound,
            "campaign_active_days": n_c
        }

    logs.append(f"Workstation 2: Successfully generated Bayesian smoothed forecasts.")
    return {"anamoly_logs": logs, "forecast_results": forecasts}


def workstation_3_allocator(state: ForeCastingState, horizon: int, budget_pool: float) -> Dict[str, Any]:
    """
    Node 3: ROAS-Constrained Portfolio Optimization Simulator.
    Fixes spend dilution by using campaign-specific lifespans and introduces structural ROAS guardrails.
    """
    master_df = state["cleaned_data"].get("master_df")
    forecasts = state["forecast_results"]
    
    if master_df is None or not forecasts:
        return {"anamoly_logs": ["Workstation 3 [HALT]: Critical data streams are missing."], "simulation_results": {}}
    
    hierarchy_summary = master_df.groupby(['channel', 'campaign_name']).agg(
        total_spend=('spend', 'sum'),
        total_revenue=('revenue', 'sum')
    ).reset_index()
    
    global_days = len(master_df['date'].unique())
    global_daily_spend_prior = master_df['spend'].sum() / (len(hierarchy_summary) * global_days) if len(hierarchy_summary) > 0 else 1.0
    
    allocation_scores = {}
    roas_bounds_matrix = {}
    total_score = 0
    max_roas_cap = 10.0  # Prevents volatile micro-campaigns from throwing off budget distribution weights
    shrinkage_half_life = 14
    
    for _, row in hierarchy_summary.iterrows():
        ch, camp = row['channel'], row['campaign_name']
        key = f"{ch} | {camp}"
        if key not in forecasts:
            continue
            
        pred = forecasts[key]
        n_c = pred["campaign_active_days"]
        
        hist_spend = row['total_spend']
        hist_rev = row['total_revenue']
        
        # Base historical ROAS calculation
        base_roas = (hist_rev / hist_spend) if hist_spend > 0 else 1.0
        base_roas = min(base_roas, max_roas_cap)
        
        # --- THE FIX: Calculate actual, non-diluted campaign run-rates ---
        raw_daily_spend = hist_spend / n_c
        w_c = n_c / (n_c + shrinkage_half_life)
        smoothed_daily_spend = (w_c * raw_daily_spend) + ((1 - w_c) * global_daily_spend_prior)
        
        # Scale run rate to match the forecasting horizon
        avg_spend_per_horizon = max(1.0, smoothed_daily_spend * horizon)
        
        # Calculate expected ROAS boundaries
        expected_roas = pred["expected_revenue"] / avg_spend_per_horizon
        lower_roas = pred["lower_bound_revenue"] / avg_spend_per_horizon
        upper_roas = pred["upper_bound_revenue"] / avg_spend_per_horizon
        
        # Apply the ROAS Ceiling Guardrail
        if expected_roas > max_roas_cap:
            expected_roas = max_roas_cap
            lower_roas = min(lower_roas, max_roas_cap)
            upper_roas = min(upper_roas, max_roas_cap)
            
        roas_bounds_matrix[key] = {
            "expected_roas": expected_roas,
            "lower_roas": lower_roas,
            "upper_roas": upper_roas,
            "avg_spend_per_horizon": avg_spend_per_horizon
        }
        
        # Volatility index
        uncertainty_penalty = (pred["upper_bound_revenue"] - pred["lower_bound_revenue"]) / (pred["expected_revenue"] + 1)
        allocation_score = base_roas * (1.0 / (1.0 + uncertainty_penalty))
        
        allocation_scores[key] = allocation_score
        total_score += allocation_score

    portfolio_allocations = {}
    blended_revenue_projection = 0.0
    
    for key, score in allocation_scores.items():
        share = score / total_score if total_score > 0 else (1.0 / len(allocation_scores))
        allocated_funds = budget_pool * share
        final_allocated_funds = max(budget_pool * 0.005, allocated_funds)
        
        portfolio_allocations[key] = final_allocated_funds
        blended_revenue_projection += final_allocated_funds * roas_bounds_matrix[key]["expected_roas"]

    sim_outputs = {
        "portfolio_allocations": portfolio_allocations,
        "roas_bounds_matrix": roas_bounds_matrix,
        "blended_revenue_projection": blended_revenue_projection
    }
    return {"anamoly_logs": ["Workstation 3: Dynamic campaign-specific allocations complete."], "simulation_results": sim_outputs}


def workstation_4_llm_narrative(state: ForeCastingState, horizon: int, budget_pool: float) -> Dict[str, Any]:
    """
    Node 4: LLM Narrative Generator Engine.
    Leverages Gemini 2.5 Flash with strict formatting guardrails to prevent LaTeX formatting errors.
    """
    sim_res = state.get("simulation_results", {})
    allocations = sim_res.get("portfolio_allocations", {})
    roas_matrix = sim_res.get("roas_bounds_matrix", {})
    blended_rev = sim_res.get("blended_revenue_projection", 0.0)
    
    api_key = os.getenv("GEMINI_API_KEY") 
    if not api_key:
        return {
            "anamoly_logs": ["Workstation 4 [WARN]: Gemini API Key missing in environment variables."],
            "executive_summary": "⚠️ **LLM Summary Unavailable**: Please verify your configuration contains a valid API key to unlock AI reasoning generation."
        }
        
    metrics_context = f"""
    --- OPERATIONAL METRICS CONTEXT ---
    Planning Horizon Scenario: {horizon} Days
    Total Portfolio Budget Pool: USD {budget_pool:,.2f}
    Projected Blended Portfolio Revenue: USD {blended_rev:,.2f}
    Projected Blended Portfolio ROAS: {blended_rev / budget_pool if budget_pool > 0 else 0:.2f}x
    
    Granular Campaign Distributions:
    """
    for key, allocated_funds in allocations.items():
        metrics_context += f"\n- Campaign Category: {key} | Allocated Budget: USD {allocated_funds:,.2f} | Target Expected ROAS: {roas_matrix[key]['expected_roas']:.2f}x (Lower Range Limit: {roas_matrix[key]['lower_roas']:.2f}x, Upper Range Limit: {roas_matrix[key]['upper_roas']:.2f}x)"

    # Formulate System Directed Prompt with strict escaping/formatting rules
    prompt = f"""
    You are an expert Chief Marketing Officer (CMO) and Data Science lead. Review the following multi-channel forecasting analytics payload and write an executive briefing.
    
    {metrics_context}
    
    Your briefing MUST include:
    1. A strategic explanation of why these specific budget shares were assigned based on campaign performance and volatility.
    2. A crisp analysis highlighting the top channel opportunity vs the channel with the highest volatility risk.
    3. Actionable tactical recommendations for an ad operations team to execute over the next {horizon} days.
    
    CRITICAL FORMATTING RULES:
    - NEVER use a raw dollar sign ($) followed immediately by numbers. This completely breaks our Markdown parser by triggering unintentional LaTeX rendering blocks. 
    - Always use the text abbreviation "USD " before writing currency numbers (e.g., write "USD 50,000" or "USD 16,742,513.82").
    - Ensure clean text formatting with standard bullet points. Jump directly into the insights.
    """
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return {
            "anamoly_logs": ["Workstation 4: Executive synthesis generated through Gemini successfully with safe string filters."],
            "executive_summary": response.text
        }
    except Exception as e:
        return {
            "anamoly_logs": [f"Workstation 4 [EXCEPT]: LLM execution fault: {str(e)}"],
            "executive_summary": f"❌ **LLM Node Execution Fault**: {str(e)}"
        }


# -----------------------------------------------------------------
# CENTRAL FLOW RUNTIME CONTROLLER
# -----------------------------------------------------------------
if google_file or meta_file or bing_file:
    
    current_state: ForeCastingState = {
        "anamoly_logs": [],
        "raw_data_status": "uninitialized",
        "cleaned_data": {},
        "forecast_results": {},
        "simulation_results": {},
        "executive_summary": ""
    }
    
    # 1. Fire Node 1 Ingestion
    node1_update = workstation_1_ingest(current_state, google_file, meta_file, bing_file)
    current_state["anamoly_logs"] += node1_update["anamoly_logs"]
    current_state["raw_data_status"] = node1_update["raw_data_status"]
    current_state["cleaned_data"] = node1_update["cleaned_data"]
    
    if current_state["raw_data_status"] == "success":
        
        # 2. Fire Node 2 Forecasting
        node2_update = workstation_2_forecaster(current_state, planning_horizon)
        current_state["anamoly_logs"] += node2_update["anamoly_logs"]
        current_state["forecast_results"] = node2_update["forecast_results"]
        
        # 3. Fire Node 3 Allocation Simulation
        node3_update = workstation_3_allocator(current_state, planning_horizon, future_target_budget)
        current_state["anamoly_logs"] += node3_update["anamoly_logs"]
        current_state["simulation_results"] = node3_update["simulation_results"]
        
        # 4. Fire Node 4 LLM Generation Layer
        with st.spinner("🤖 Multi-Agent Engine consulting Gemini for Strategic Insights..."):
            node4_update = workstation_4_llm_narrative(current_state, planning_horizon, future_target_budget)
            current_state["anamoly_logs"] += node4_update["anamoly_logs"]
            current_state["executive_summary"] = node4_update["executive_summary"]
        
        sim_res = current_state["simulation_results"]
        blended_portfolio_roas = sim_res["blended_revenue_projection"] / future_target_budget if future_target_budget > 0 else 0

        # --- EXECUTIVE SCORECARD METRICS DISPLAY ---
        st.write("## 📌 Executive Optimization Dashboard")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Planning Target Window", f"{planning_horizon} Days")
        m_col2.metric("Projected Blended Portfolio Revenue", f"${sim_res['blended_revenue_projection']:,.2f}")
        m_col3.metric("Projected Blended Portfolio ROAS", f"{blended_portfolio_roas:.2f}x")
        
        st.write("---")
        
        # --- PORTFOLIO VISUALIZATION DESIGN PANELS ---
        st.write("### 📊 Advanced Performance & Budget Allocation BI Workspace")
        
        chart_records = []
        for key, budget in sim_res["portfolio_allocations"].items():
            ch = current_state["forecast_results"][key]["channel"]
            camp = current_state["forecast_results"][key]["campaign"]
            chart_records.append({
                "Channel": ch,
                "Campaign": camp,
                "Budget": budget,
                "Expected_ROAS": sim_res["roas_bounds_matrix"][key]["expected_roas"],
                "Min_ROAS": sim_res["roas_bounds_matrix"][key]["lower_roas"],
                "Max_ROAS": sim_res["roas_bounds_matrix"][key]["upper_roas"]
            })
            
        ui_df = pd.DataFrame(chart_records)
        
        # 1. PIE CHART: Occupies the upper section cleanly
        st.markdown("##### 🍕 Aggregate Portfolio Capital Allocations")
        channel_summary_chart = ui_df.groupby("Channel")["Budget"].sum().reset_index()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=channel_summary_chart["Channel"], 
            values=channel_summary_chart["Budget"], 
            hole=.45,
            marker=dict(colors=["#1F77B4", "#2CA02C", "#FF7F0E"]), # Unified professional palette
            hoverinfo="label+percent+value"
        )])
        fig_pie.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.write("---")

        # 2. SCATTER PLOT: Occupies a full wide block to maximize scannability
        st.markdown("##### 🎯 Efficiency Frontier Workspace (Granular Campaign Positioning)")
        
        fig_scatter = go.Figure()
        
        # Define uniform channel mappings for colors to match the pie chart perfectly
        channel_colors = {
            "Google_Ads": "#1F77B4",
            "Meta_Ads": "#2CA02C",
            "Bing_Ads": "#FF7F0E"
        }
        
        # Map campaign bubble sizing dynamically based on relative ROAS performance 
        # (Add a baseline minimum size so lower performance values don't vanish entirely)
        bubble_sizes = ui_df["Expected_ROAS"].apply(lambda x: max(8, min(x * 3.5, 35)))
        
        for channel_name, group in ui_df.groupby("Channel"):
            fig_scatter.add_trace(go.Scatter(
                x=group["Budget"],
                y=group["Expected_ROAS"],
                mode='markers', # Dropped raw +text to instantly kill the text-overlap cloud
                name=channel_name.replace('_', ' '),
                marker=dict(
                    size=group["Expected_ROAS"].apply(lambda x: max(8, min(x * 3.5, 35))),
                    color=channel_colors.get(channel_name, "#7F7F7F"),
                    line=dict(width=1, color="white"),
                    opacity=0.85
                ),
                customdata=np.stack((group['Campaign'], group['Min_ROAS'], group['Max_ROAS']), axis=-1),
                hovertemplate="""
                <b>Campaign:</b> %{customdata[0]}<br>
                <b>Channel:</b> """ + channel_name.replace('_', ' ') + """<br>
                <b>Allocated Budget:</b> USD %{x:,.2f}<br>
                <b>Expected ROAS:</b> %{y:.2f}x<br>
                <b>Confidence Interval:</b> %{customdata[1]:.2f}x - %{customdata[2]:.2f}x<br>
                <extra></extra>
                """
            ))
            
        # Draw a horizontal target baseline matching portfolio blended average efficiency
            # Draw a horizontal target baseline matching portfolio blended average efficiency
        fig_scatter.add_hline(
            y=blended_portfolio_roas,
            line_dash="dash",
            line_color="#d32f2f",
            line_width=1.5,
            annotation_text=f"Blended ROAS Baseline ({blended_portfolio_roas:.2f}x)",
            annotation_position="top right",
            annotation_font=dict(color="#d32f2f", size=11)
        )
        
        fig_scatter.update_layout(
            margin=dict(t=15, b=15, l=10, r=10),
            height=450, # Tall, wide, and easily scannable
            xaxis_title="Allocated Budget Allocation Pool (USD)",
            yaxis_title="Expected Probabilistic ROAS Range",
            xaxis=dict(gridcolor="#f1f3f5", zeroline=False),
            yaxis=dict(gridcolor="#f1f3f5", zeroline=False),
            plot_bgcolor="white",
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- GRANULAR RANKING DATA TABLE MATRIX ---
        st.write("### 🔍 Granular Campaign Hierarchy Analytics Range Matrix")
        formatted_ui_df = ui_df.copy()
        formatted_ui_df = formatted_ui_df.rename(columns={
            "Budget": "Allocated Budget ($)",
            "Expected_ROAS": "Expected ROAS Range",
            "Min_ROAS": "Lower Bound ROAS Limit",
            "Max_ROAS": "Upper Bound ROAS Limit"
        })
        st.dataframe(
            formatted_ui_df.sort_values(by="Allocated Budget ($)", ascending=False).reset_index(drop=True), 
            use_container_width=True
        )

        # --- WORKSTATION 4 AI GENERATED NARRATIVE PANEL ---
        st.write("---")
        st.write("### 🤖 Agentic Executive Insights Briefing")
        st.markdown(current_state["executive_summary"])

        # --- SYSTEM MONITOR PIPELINE DIAGNOSTIC LOGS ---
        with st.expander("🛠️ Core Multi-Agent System Pipeline Logs"):
            for log in current_state["anamoly_logs"]:
                st.caption(f"⚙️ {log}")
    else:
        st.error("Data processing collapse: Master matrix returned zero uniform rows.")
else:
    st.info("👋 Welcome Engineer! Please upload at least one channel tracking dataset (.csv) in the left configuration control panel to activate the forecasting pipelines.")
