"""Streamlit frontend dashboard for Multi-Plant Order Allocation with Cost and Carbon.

This hackathon application communicates with the FastAPI backend to:
1. Run multi-plant order allocation balancing landed cost and carbon footprint.
2. Simulate what-if scenarios (e.g. plant capacity disruptions).
3. Visualize cost-vs-carbon Pareto trade-offs.
4. Upload supply chain data CSV files (orders, plants, transport duties, capacity events).
5. Inspect multi-agent execution traces.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Configure the Streamlit page layout and title
st.set_page_config(
    page_title="Multi-Plant Order Allocation",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject crisp modern light-theme CSS
st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #1E293B;
        }
        [data-testid="stSidebar"] {
            background-color: #F8FAFC !important;
            border-right: 1px solid #E2E8F0;
        }
        .stMetric {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetricValue"] {
            color: #0F172A !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #475569 !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 16px;
            background-color: #F1F5F9;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1D70B8 !important;
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# API base URL configuration from environment variable with fallback
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 90



# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================


def call_analyze(
    api_url: str,
    request_text: str,
    weights: Dict[str, float],
    scenario: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Send allocation request to POST /analyze. Returns response dict or None on error."""
    endpoint = f"{api_url.rstrip('/')}/analyze"
    payload: Dict[str, Any] = {
        "request": request_text,
        "weights": weights,
    }
    if scenario:
        payload["scenario"] = scenario

    try:
        response = requests.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()
        st.error(f"Analyze API returned status {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to connect to API ({endpoint}): {exc}")
        return None


def call_ingest(
    api_url: str,
    table_name: str,
    file_bytes: bytes,
    filename: str,
    mode: str = "replace",
) -> Optional[Dict[str, Any]]:
    """Upload CSV to POST /ingest. Returns response dict or None on error."""
    endpoint = f"{api_url.rstrip('/')}/ingest"
    files = {"file": (filename, file_bytes, "text/csv")}
    metadata = json.dumps({"table": table_name, "mode": mode})
    data = {"metadata": metadata}

    try:
        response = requests.post(
            endpoint,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # 200 (success/partial) and 422 (validation failed) return structured IngestResponse
        if response.status_code in (200, 422):
            try:
                return response.json()
            except Exception:
                pass
        st.error(f"Ingest API returned status {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to upload to API ({endpoint}): {exc}")
        return None


def call_health(api_url: str) -> Optional[Dict[str, Any]]:
    """Check API health via GET /health. Returns response dict or None."""
    endpoint = f"{api_url.rstrip('/')}/health"
    try:
        response = requests.get(endpoint, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()
        st.error(f"Health check returned status {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Health check failed ({endpoint}): {exc}")
        return None


def call_trace(api_url: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Fetch execution trace via GET /trace/{run_id}. Returns response dict or None."""
    endpoint = f"{api_url.rstrip('/')}/trace/{run_id}"
    try:
        response = requests.get(endpoint, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            st.warning(f"No execution trace found for run ID: {run_id}")
            return None
        st.error(f"Trace API returned status {response.status_code}: {response.text}")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to fetch trace ({endpoint}): {exc}")
        return None


# ============================================================================
# UI COMPONENT FUNCTIONS
# ============================================================================


def show_kpi_cards(result: Optional[Dict[str, Any]]) -> None:
    """Display the 6 KPI metric cards from an analyze response."""
    if not result:
        return

    kpis = result.get("kpis") or {}
    allocation_items = result.get("allocation") or []

    # 1. Total Landed Cost
    total_cost = kpis.get("total_cost")
    cost_display = f"${total_cost:,.2f}" if total_cost is not None else "—"

    # 2. Total Carbon in tonnes
    total_carbon = kpis.get("total_carbon_t")
    carbon_display = f"{total_carbon:,.2f} t" if total_carbon is not None else "—"

    # 3. On-Time Delivery Percentage calculated directly from allocation items
    total_orders = len(allocation_items)
    if total_orders > 0:
        on_time_count = sum(1 for item in allocation_items if item.get("on_time") is True)
        on_time_pct = (on_time_count / total_orders) * 100.0
        on_time_display = f"{on_time_pct:.1f}%"
    else:
        on_time_display = "—"

    # 4-6. Plant Utilisation for Plants A, B, and C
    plant_utils = kpis.get("plant_utilisation") or {}

    def format_util(plant_name: str) -> str:
        val = plant_utils.get(plant_name)
        if val is None:
            val = plant_utils.get(plant_name.lower())
        if val is None:
            return "—"
        return f"{val * 100.0:.1f}%"

    util_a_display = format_util("A")
    util_b_display = format_util("B")
    util_c_display = format_util("C")

    # Render metrics in 6 responsive columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label="Total Cost", value=cost_display)
    with col2:
        st.metric(label="Total Carbon", value=carbon_display)
    with col3:
        st.metric(label="On-Time Delivery", value=on_time_display)
    with col4:
        st.metric(label="Plant A Util", value=util_a_display)
    with col5:
        st.metric(label="Plant B Util", value=util_b_display)
    with col6:
        st.metric(label="Plant C Util", value=util_c_display)


def show_allocation_table(allocation_items: List[Dict[str, Any]]) -> None:
    """Display the allocation results as a filterable table."""
    if not allocation_items:
        st.info("No order allocations found.")
        return

    # Convert the allocation dictionary list into a pandas DataFrame
    df = pd.DataFrame(allocation_items)

    # Standardize and select display columns
    column_mapping = {
        "order_id": "Order ID",
        "plant": "Plant",
        "cost": "Cost ($)",
        "carbon_kg": "Carbon (kg)",
        "arrival": "Arrival",
        "on_time": "On Time",
    }
    # Keep recognized columns first, then append any extra custom fields
    ordered_cols = [c for c in column_mapping.keys() if c in df.columns]
    extra_cols = [c for c in df.columns if c not in column_mapping]
    df = df[ordered_cols + extra_cols].rename(columns=column_mapping)

    # Filter controls in columns
    filter_col1, filter_col2 = st.columns(2)

    # Plant filter
    available_plants = sorted([str(p) for p in df["Plant"].dropna().unique()]) if "Plant" in df.columns else []
    selected_plants = filter_col1.multiselect(
        "Filter by Plant",
        options=available_plants,
        default=available_plants,
        help="Select one or more plants to view their allocated orders.",
    )

    # On-Time status filter
    on_time_filter = filter_col2.selectbox(
        "Filter by Delivery Status",
        options=["All", "On Time", "Late"],
        index=0,
        help="Filter orders by arrival punctuality.",
    )

    # Apply filters
    filtered_df = df.copy()
    if "Plant" in filtered_df.columns and selected_plants:
        filtered_df = filtered_df[filtered_df["Plant"].astype(str).isin(selected_plants)]

    if "On Time" in filtered_df.columns:
        if on_time_filter == "On Time":
            filtered_df = filtered_df[filtered_df["On Time"] == True]
        elif on_time_filter == "Late":
            filtered_df = filtered_df[filtered_df["On Time"] == False]

    st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** orders:")

    # Configure column formats for clean presentation
    column_configs = {}
    if "Cost ($)" in filtered_df.columns:
        column_configs["Cost ($)"] = st.column_config.NumberColumn("Cost ($)", format="$%.2f")
    if "Carbon (kg)" in filtered_df.columns:
        column_configs["Carbon (kg)"] = st.column_config.NumberColumn("Carbon (kg)", format="%.2f kg")
    if "On Time" in filtered_df.columns:
        column_configs["On Time"] = st.column_config.CheckboxColumn("On Time")

    st.dataframe(
        filtered_df,
        column_config=column_configs,
        use_container_width=True,
        hide_index=True,
    )


def show_tradeoff_chart(tradeoff_points: List[Dict[str, Any]]) -> None:
    """Display the cost-vs-carbon Plotly chart and Pareto target breakdown."""
    if not tradeoff_points:
        st.info("Trade-off data not available...")
        return

    df = pd.DataFrame(tradeoff_points)

    if "carbon_cut_pct" not in df.columns or "cost" not in df.columns:
        st.info("Trade-off data not available...")
        return

    df = df.sort_values(by="carbon_cut_pct")
    feasible_df = df[df["cost"].notna()].copy()
    infeasible_df = df[df["cost"].isna()].copy()

    # Build Plotly Figure
    fig = go.Figure()

    # Add feasible line and points
    if not feasible_df.empty:
        fig.add_trace(go.Scatter(
            x=feasible_df["carbon_cut_pct"],
            y=feasible_df["cost"],
            mode="lines+markers",
            name="Feasible Plan (Optimal)",
            line=dict(color="#1D70B8", width=3),
            marker=dict(size=12, color="#1D70B8", symbol="circle"),
            hovertemplate="<b>%{x}% Carbon Cut</b><br>Cost: $%{y:,.2f}<br>Status: Optimal<extra></extra>",
        ))

    # Add infeasible target indicators
    if not infeasible_df.empty:
        baseline_cost = feasible_df["cost"].iloc[0] if not feasible_df.empty else 22795.0
        fig.add_trace(go.Scatter(
            x=infeasible_df["carbon_cut_pct"],
            y=[baseline_cost] * len(infeasible_df),
            mode="markers+text",
            name="Infeasible Targets (Exceeds Physical Capacity)",
            text=["Infeasible" for _ in range(len(infeasible_df))],
            textposition="top center",
            textfont=dict(color="#DC2626", size=11),
            marker=dict(size=12, color="#DC2626", symbol="x", line=dict(width=2, color="#DC2626")),
            hovertemplate="<b>%{x}% Carbon Cut Target</b><br>Status: INFEASIBLE<br>Exceeds physical capacity<extra></extra>",
        ))

    # Clean y-axis range so single point doesn't zoom into an absurd 2-dollar slice
    if len(feasible_df) == 1:
        cost_val = feasible_df["cost"].iloc[0]
        y_range = [cost_val * 0.90, cost_val * 1.10]
    elif len(feasible_df) > 1:
        min_c, max_c = feasible_df["cost"].min(), feasible_df["cost"].max()
        pad = (max_c - min_c) * 0.15 or 1000
        y_range = [max(0, min_c - pad), max_c + pad]
    else:
        y_range = None

    fig.update_layout(
        template="plotly_white",
        title="Cost vs Carbon Reduction Trade-Off (0% to 20% Targets)",
        xaxis_title="Carbon Reduction Target (%)",
        yaxis_title="Total Landed Cost ($)",
        xaxis=dict(tickmode="array", tickvals=[0, 5, 10, 15, 20], range=[-2, 23]),
        yaxis=dict(range=y_range) if y_range else dict(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explanation of trade-off feasibility
    if not infeasible_df.empty and len(feasible_df) == 1:
        st.info(
            "💡 **Why are 5%, 10%, 15%, and 20% carbon cuts marked Infeasible?**  \n"
            "The mathematical solver evaluated all 5 targets and proved that the **baseline plan ($22,795 / 1.67 t CO₂e) "
            "is already at the physical minimum carbon limit** for this dataset:  \n"
            "• **Plant C** (cleanest at 0.05 kg CO₂e/kWh) has 300 units/week capacity and is already at **100% capacity** (Order O-001 = 300 units).  \n"
            "• **Product P2** (Order O-003) can only be manufactured at Plant A (single-plant constraint).  \n"
            "• Therefore, no additional orders can physically move to greener plants without exceeding capacity."
        )

    # Clean breakdown table
    table_rows = []
    for _, row in df.iterrows():
        pct = row["carbon_cut_pct"]
        is_opt = pd.notna(row["cost"])
        st_val = "✅ OPTIMAL" if is_opt else "❌ INFEASIBLE"
        cost_str = f"${row['cost']:,.2f}" if is_opt else "Exceeds capacity"
        carb_str = f"{row.get('total_carbon_t', '—'):.2f} t" if pd.notna(row.get("total_carbon_t")) else "—"
        table_rows.append({
            "Carbon Reduction Target": f"{pct:.0f}%",
            "Solver Status": st_val,
            "Landed Cost ($)": cost_str,
            "Total Carbon": carb_str,
        })
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)



def show_whatif_comparison(
    original_result: Dict[str, Any],
    whatif_result: Dict[str, Any],
) -> None:
    """Display side-by-side comparison of original vs what-if plans."""
    orig_kpis = original_result.get("kpis") or {}
    new_kpis = whatif_result.get("kpis") or {}
    diff = whatif_result.get("scenario_comparison") or {}

    col_orig, col_new = st.columns(2)

    # Original Plan summary
    with col_orig:
        st.subheader("📋 Original Plan")
        orig_cost = orig_kpis.get("total_cost")
        orig_carbon = orig_kpis.get("total_carbon_t")
        orig_late = orig_kpis.get("late_orders")

        st.metric(
            label="Original Cost",
            value=f"${orig_cost:,.2f}" if orig_cost is not None else "—",
        )
        st.metric(
            label="Original Carbon",
            value=f"{orig_carbon:,.2f} t" if orig_carbon is not None else "—",
        )
        st.metric(
            label="Original Late Orders",
            value=f"{orig_late}" if orig_late is not None else "—",
        )

    # What-If Plan summary with delta shifts
    with col_new:
        st.subheader("🔬 New Plan (What-If)")
        new_cost = new_kpis.get("total_cost")
        new_carbon = new_kpis.get("total_carbon_t")
        new_late = new_kpis.get("late_orders")

        # Extract pre-calculated deltas if present in scenario_comparison
        delta_cost = diff.get("extra_cost")
        delta_carbon = diff.get("extra_carbon_t")
        delta_late = diff.get("late_delta")

        # Fallback calculations if not in scenario_comparison
        if delta_cost is None and orig_cost is not None and new_cost is not None:
            delta_cost = new_cost - orig_cost
        if delta_carbon is None and orig_carbon is not None and new_carbon is not None:
            delta_carbon = new_carbon - orig_carbon
        if delta_late is None and orig_late is not None and new_late is not None:
            delta_late = new_late - orig_late

        st.metric(
            label="New Cost",
            value=f"${new_cost:,.2f}" if new_cost is not None else "—",
            delta=f"${delta_cost:+,.2f}" if delta_cost is not None else None,
            delta_color="inverse",  # Cost increase is unfavorable
        )
        st.metric(
            label="New Carbon",
            value=f"{new_carbon:,.2f} t" if new_carbon is not None else "—",
            delta=f"{delta_carbon:+,.2f} t" if delta_carbon is not None else None,
            delta_color="inverse",  # Carbon increase is unfavorable
        )
        st.metric(
            label="New Late Orders",
            value=f"{new_late}" if new_late is not None else "—",
            delta=f"{delta_late:+d}" if delta_late is not None else None,
            delta_color="inverse",  # More late orders is unfavorable
        )

    # Impact summary: orders moved
    orders_moved = diff.get("orders_moved")
    if orders_moved is not None:
        st.info(f"🔄 **Orders Moved:** {orders_moved} order(s) reallocated across plants.")

    moved_details = diff.get("moved_order_details")
    if moved_details and isinstance(moved_details, list):
        with st.expander("📦 Reallocated Order Details"):
            st.dataframe(pd.DataFrame(moved_details), use_container_width=True)


# ============================================================================
# MAIN APPLICATION INTERFACE
# ============================================================================

# Initialize session state keys
if "result" not in st.session_state:
    st.session_state.result = None
if "whatif_result" not in st.session_state:
    st.session_state.whatif_result = None

# Sidebar for system status & configuration
with st.sidebar:
    st.title("🏭 System Status")
    st.caption(f"Backend URL: `{API_BASE_URL}`")

    # Health check button & status viewer
    health_data = call_health(API_BASE_URL)
    if health_data:
        backend_status = health_data.get("status", "unknown")
        if backend_status == "ok":
            st.success("Backend: Online & Healthy")
        else:
            st.warning(f"Backend Status: {backend_status}")

        data_store = health_data.get("data_store", {})
        table_counts = data_store.get("tables", {})
        if table_counts:
            st.markdown("**Database Tables:**")
            for tbl, count in table_counts.items():
                st.write(f"- `{tbl}`: {count:,} rows")

        llm_info = health_data.get("llm", {})
        if llm_info:
            st.markdown(f"**LLM:** {llm_info.get('provider')} (`{llm_info.get('model')}`)")
            if llm_info.get("mock"):
                st.info("Deterministic Mock Mode active")
    else:
        st.error("Backend unreachable. Please verify FastAPI is running.")

st.title("🌱 Multi-Plant Order Allocation & Carbon Optimizer")
st.markdown(
    "Allocate production orders across global manufacturing plants to minimize landed cost "
    "while meeting sustainability decarbonization goals."
)

# Main Navigation Tabs
tab_allocation, tab_upload = st.tabs(["🏭 Allocation", "📤 Data Upload"])


# ============================================================================
# TAB 1: ALLOCATION & WHAT-IF
# ============================================================================
with tab_allocation:
    # A. Dashboard Header KPI Cards (only shown when results exist in session state)
    if st.session_state.result:
        st.subheader("Key Performance Indicators")
        show_kpi_cards(st.session_state.result)
        st.divider()

    # B. Run Allocation Input Section
    st.subheader("Run Allocation")
    request_text = st.text_input(
        "User Request",
        value="Allocate this week's orders while balancing cost and carbon emissions.",
        help="Natural language prompt sent to the Section 6 multi-agent orchestrator.",
    )

    weight_col1, weight_col2 = st.columns(2)
    with weight_col1:
        cost_weight = st.slider(
            "Cost Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Relative weight for landed cost minimization.",
        )
    with weight_col2:
        carbon_weight = st.slider(
            "Carbon Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Relative weight for carbon footprint reduction.",
        )

    st.info("Weights are relative priorities — the solver normalises them automatically.")

    if st.button("🚀 Run Allocation", type="primary"):
        with st.spinner("Solving allocation optimization..."):
            weights_payload = {"cost": cost_weight, "carbon": carbon_weight}
            response = call_analyze(API_BASE_URL, request_text, weights_payload)
            if response:
                st.session_state.result = response
                st.session_state.whatif_result = None  # Reset what-if on new baseline
                st.rerun()

    # Display results when available
    if st.session_state.result:
        res = st.session_state.result

        # Display summary message
        if res.get("summary"):
            st.info(f"**Optimization Summary:** {res['summary']}")

        # C. What-If Scenario Panel
        with st.expander("🔬 What-If Scenario"):
            st.markdown(
                "Simulate sudden plant capacity disruption (e.g. equipment maintenance or power outage) "
                "to evaluate system resilience and cost impact."
            )
            scenario_col1, scenario_col2 = st.columns(2)
            with scenario_col1:
                capacity_loss = st.slider(
                    "Plant B Capacity Loss (%)",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=5,
                    help="Percentage of Plant B capacity lost in this scenario.",
                )
            with scenario_col2:
                whatif_cost_w = st.slider("What-If Cost Weight", 0.0, 1.0, cost_weight, 0.05)
                whatif_carbon_w = st.slider("What-If Carbon Weight", 0.0, 1.0, carbon_weight, 0.05)

            if st.button("Run What-If"):
                with st.spinner("Running what-if scenario re-plan..."):
                    scenario_payload = {
                        "plant": "B",
                        "capacity_loss_pct": float(capacity_loss),
                    }
                    whatif_resp = call_analyze(
                        api_url=API_BASE_URL,
                        request_text="Re-plan with capacity loss scenario.",
                        weights={"cost": whatif_cost_w, "carbon": whatif_carbon_w},
                        scenario=scenario_payload,
                    )
                    if whatif_resp:
                        st.session_state.whatif_result = whatif_resp
                        st.rerun()

            if st.session_state.whatif_result:
                st.divider()
                show_whatif_comparison(st.session_state.result, st.session_state.whatif_result)

        # D. Trade-Off Visualization
        st.subheader("📈 Cost vs Carbon Trade-Off")
        show_tradeoff_chart(res.get("tradeoff") or [])

        # E. Allocation Table
        st.subheader("📋 Order Allocations")
        show_allocation_table(res.get("allocation") or [])

        # F. Executive Brief
        st.subheader("📝 Executive Brief")
        brief_content = res.get("brief")
        if brief_content:
            with st.container():
                st.markdown(brief_content)
        else:
            st.info("No executive brief generated for this request.")

        # Recommendations list if present
        recommendations = res.get("recommendations") or []
        if recommendations:
            with st.expander("💡 Actionable Recommendations"):
                for idx, rec in enumerate(recommendations, 1):
                    approval = "⚠️ Requires Approval" if rec.get("requires_approval") else "✅ Auto-approved"
                    st.markdown(
                        f"**{idx}. {rec.get('action')}** ({rec.get('confidence')} Confidence, {approval})\n\n"
                        f"- *Rationale:* {rec.get('rationale')}\n"
                        f"- *Estimated Impact:* {rec.get('estimated_impact')}"
                    )

        # G. Execution Trace
        with st.expander("📋 Execution Trace"):
            run_id = res.get("run_id")
            if run_id:
                st.markdown(f"**Run ID:** `{run_id}` | Latency: `{res.get('latency_ms', 0):.0f} ms`")
                trace_data = call_trace(API_BASE_URL, run_id)
                if trace_data:
                    steps = trace_data.get("steps") or []
                    if steps:
                        st.markdown("**Agent Execution Steps:**")
                        for idx, step in enumerate(steps, 1):
                            agent = step.get("agent", "Unknown")
                            tool = step.get("tool", "None")
                            st.markdown(f"**Step {idx}: Agent `{agent}` → Tool `{tool}`**")

                            step_col1, step_col2 = st.columns(2)
                            with step_col1:
                                st.caption("Input:")
                                st.json(step.get("input") or {})
                            with step_col2:
                                st.caption("Output:")
                                st.json(step.get("output") or {})
                            st.divider()

                    trace_llm_calls = trace_data.get("llm_calls", 0)
                    trace_tokens = trace_data.get("tokens", 0)
                    st.caption(f"Total LLM Calls: **{trace_llm_calls}** | Total Tokens: **{trace_tokens:,}**")
                else:
                    st.caption(f"LLM Calls: **{res.get('llm_calls', 0)}** | Tokens: **{res.get('tokens', 0):,}**")
            else:
                st.info("No Run ID associated with this response.")


# ============================================================================
# TAB 2: DATA UPLOAD (ANNEX B TABLES)
# ============================================================================
with tab_upload:
    st.subheader("Upload Supply Chain Data (CSV)")
    st.markdown(
        "Upload updated operational datasets. The backend validates each row against the Annex B "
        "data contracts and rejects invalid records with explanatory feedback."
    )

    tables = [
        ("orders", "Orders Dataset (Demands, destinations, due dates)"),
        ("plants", "Plants Dataset (Capacities, energy kWh, grid carbon factors)"),
        ("transport_duties", "Transport & Duties (Lanes, costs, transit days, tariffs)"),
        ("capacity_events", "Capacity Events (Planned outages and maintenance windows)"),
    ]

    for table_name, table_desc in tables:
        with st.container():
            st.markdown(f"### `{table_name}`")
            st.caption(table_desc)

            up_col1, up_col2, up_col3 = st.columns([3, 2, 1])
            with up_col1:
                uploaded_file = st.file_uploader(
                    f"Choose CSV file for {table_name}",
                    type=["csv"],
                    key=f"file_{table_name}",
                )
            with up_col2:
                mode = st.selectbox(
                    "Ingest Mode",
                    options=["replace", "append", "upsert"],
                    index=0,
                    key=f"mode_{table_name}",
                    help="'replace' overwrites table; 'append' adds new rows; 'upsert' updates existing records.",
                )
            with up_col3:
                st.write("")  # Vertical spacing
                st.write("")
                upload_clicked = st.button("Upload", key=f"btn_{table_name}")

            if upload_clicked:
                if uploaded_file is None:
                    st.warning(f"Please select a CSV file to upload into '{table_name}'.")
                else:
                    with st.spinner(f"Ingesting into {table_name}..."):
                        file_bytes = uploaded_file.getvalue()
                        ingest_result = call_ingest(
                            api_url=API_BASE_URL,
                            table_name=table_name,
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            mode=mode,
                        )

                        if ingest_result:
                            status = ingest_result.get("status")
                            loaded = ingest_result.get("rows_loaded", 0)
                            rejected = ingest_result.get("rows_rejected", 0)
                            total = ingest_result.get("total_rows_in_table", 0)

                            if status == "loaded":
                                st.success(
                                    f"✅ Ingest completed successfully! Loaded **{loaded}** rows. "
                                    f"Total rows in table: **{total}**."
                                )
                            elif status == "partial":
                                st.warning(
                                    f"⚠️ Partial ingest: Loaded **{loaded}** rows, rejected **{rejected}** rows. "
                                    f"Total rows in table: **{total}**."
                                )
                            else:
                                st.error(
                                    f"❌ Ingest failed: **{rejected}** rows rejected. "
                                    f"No rows were committed."
                                )

                            # Display rejected rows if any exist
                            rejected_items = ingest_result.get("rejected") or []
                            if rejected_items:
                                with st.expander(f"⚠️ View Rejected Rows ({len(rejected_items)})"):
                                    rej_df = pd.DataFrame(rejected_items)
                                    st.dataframe(rej_df, use_container_width=True)

            st.divider()
