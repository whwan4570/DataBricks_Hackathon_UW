# -*- coding: utf-8 -*-
"""
Weather-to-Yield Signal Detection — Streamlit dashboard
State-level yield and weather on map; anomaly detection (yield Z-score, yield–weather residual).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import urllib.request
import json
import os
import base64

BASE = Path(__file__).resolve().parent
MERGED_CSV = BASE / "merged_weather_yields.csv"

# US state order for consistent dropdown (excluding territories if needed)
STATE_ABBR = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]


COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


@st.cache_data
def load_county_geojson():
    """Load US counties GeoJSON (FIPS in feature id). Cached."""
    with urllib.request.urlopen(COUNTY_GEOJSON_URL) as r:
        return json.load(r)


@st.cache_data
def load_data():
    df = pd.read_csv(MERGED_CSV)
    df.columns = df.columns.str.strip()
    for col in ["Commodity Name", "State Name", "State Abbreviation", "County Name", "Irrigation Practice Name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["Yield Amount"] = pd.to_numeric(df["Yield Amount"], errors="coerce")
    for c in ["avg_temp_max_c", "avg_temp_min_c", "total_precipitation_mm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # FIPS: 5-digit = State Code (2) + County Code (3)
    if "State Code" in df.columns and "County Code" in df.columns:
        sc = pd.to_numeric(df["State Code"], errors="coerce").fillna(0).astype(int)
        cc = pd.to_numeric(df["County Code"], errors="coerce").fillna(0).astype(int)
        df["fips"] = sc.apply(lambda x: str(x).zfill(2)) + cc.apply(lambda x: str(x).zfill(3))
    return df


def state_yield_agg(df, crop, year, weather_var=None):
    """Aggregate by state for selected crop and year; include weather_var if provided."""
    sub = df[(df["Commodity Name"] == crop) & (df["year"] == year)].copy()
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby("State Abbreviation").agg(
        yield_mean=("Yield Amount", "mean"),
        yield_sum=("Yield Amount", "sum"),
        county_count=("County Name", "nunique"),
    ).reset_index()
    if weather_var and weather_var != "yield" and weather_var in sub.columns:
        w = sub.groupby("State Abbreviation")[weather_var].mean().reset_index()
        w = w.rename(columns={weather_var: "weather_value"})
        agg = agg.merge(w, on="State Abbreviation", how="left")
    return agg


def detect_yield_zscore(df, crop, year, threshold_std=2.0):
    """Flag counties with yield Z-score beyond threshold (by crop & year)."""
    sub = df[(df["Commodity Name"] == crop) & (df["year"] == year)].copy()
    if sub.empty or sub["Yield Amount"].std() == 0:
        return pd.DataFrame()
    mean_y = sub["Yield Amount"].mean()
    std_y = sub["Yield Amount"].std()
    sub = sub.assign(
        yield_zscore=(sub["Yield Amount"] - mean_y) / std_y,
        anomaly_type=np.where(sub["Yield Amount"] > mean_y + threshold_std * std_y, "High",
                              np.where(sub["Yield Amount"] < mean_y - threshold_std * std_y, "Low", None)),
    )
    return sub[sub["anomaly_type"].notna()].copy()


def detect_yield_weather_residual(df, crop, year, threshold_std=2.0):
    """Flag counties where yield deviates from weather-based expectation (residual Z-score)."""
    sub = df[(df["Commodity Name"] == crop) & (df["year"] == year)].copy()
    weather_cols = ["avg_temp_max_c", "avg_temp_min_c", "total_precipitation_mm"]
    valid = sub[["Yield Amount"] + weather_cols].notna().all(axis=1)
    sub = sub[valid].copy()
    if len(sub) < 10:
        return pd.DataFrame()
    X = sub[weather_cols].fillna(sub[weather_cols].median())
    y = sub["Yield Amount"]
    try:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
    except Exception:
        return pd.DataFrame()
    sub = sub.assign(
        yield_predicted=pred,
        residual=y.values - pred,
    )
    r_mean, r_std = sub["residual"].mean(), sub["residual"].std()
    if r_std == 0:
        return pd.DataFrame()
    sub["residual_z"] = (sub["residual"] - r_mean) / r_std
    sub["anomaly_type"] = np.where(
        sub["residual_z"] >= threshold_std, "High (yield above weather expectation)",
        np.where(sub["residual_z"] <= -threshold_std, "Low (yield below weather expectation)", None),
    )
    return sub[sub["anomaly_type"].notna()].copy()


@st.cache_data
def train_yield_model(df):
    """Train a yield prediction model (crop, state, year, weather -> yield). Returns (model, enc_state, enc_crop)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    feats = df[
        (df["Yield Amount"].notna()) &
        (df["State Abbreviation"].notna()) &
        (df["Commodity Name"].notna())
    ].copy()
    for c in ["avg_temp_max_c", "avg_temp_min_c", "total_precipitation_mm"]:
        if c in feats.columns:
            feats[c] = pd.to_numeric(feats[c], errors="coerce")
    feats = feats.dropna(subset=["Yield Amount", "State Abbreviation", "Commodity Name", "year"])
    if feats.shape[0] < 50:
        return None, None, None
    enc_s = LabelEncoder()
    enc_c = LabelEncoder()
    feats["_state"] = enc_s.fit_transform(feats["State Abbreviation"].astype(str))
    feats["_crop"] = enc_c.fit_transform(feats["Commodity Name"].astype(str))
    weather = ["avg_temp_max_c", "avg_temp_min_c", "total_precipitation_mm"]
    use_w = [x for x in weather if x in feats.columns]
    X = feats[["_state", "_crop", "year"] + use_w].fillna(feats[use_w].median() if use_w else 0)
    y = feats["Yield Amount"]
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, enc_s, enc_c


def predict_yield(model, enc_state, enc_crop, crop, state_abbr, year, temp_max=None, temp_min=None, precip=None, df_median=None):
    """Single prediction. Weather defaults to df_median if provided."""
    if model is None or enc_state is None or enc_crop is None:
        return None
    try:
        s = enc_state.transform([state_abbr])[0]
        c = enc_crop.transform([crop])[0]
    except Exception:
        return None
    if df_median is not None:
        tmax = temp_max if temp_max is not None else df_median.get("avg_temp_max_c", 20)
        tmin = temp_min if temp_min is not None else df_median.get("avg_temp_min_c", 10)
        pr = precip if precip is not None else df_median.get("total_precipitation_mm", 5000)
    else:
        tmax = temp_max if temp_max is not None else 20
        tmin = temp_min if temp_min is not None else 10
        pr = precip if precip is not None else 5000
    X = np.array([[s, c, year, tmax, tmin, pr]])
    return float(model.predict(X)[0])


def render_chat():
    """Left panel: LLM chatbot placeholder. Set OPENAI_API_KEY to use OpenAI."""
    st.subheader("Chat")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about yield, weather, or data..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = _chat_reply(prompt)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


def _chat_reply(prompt: str) -> str:
    """Generate reply. Uses OpenAI if OPENAI_API_KEY set; else placeholder."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return r.choices[0].message.content or "No response."
        except Exception as e:
            return f"API error: {e}. Using placeholder reply."
    return (
        "This is a placeholder. Set `OPENAI_API_KEY` in your environment to connect an LLM. "
        "You can ask about yield by crop/state/year, weather effects, or anomaly detection."
    )


PAGE_BG_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 25%, #f8fafc 60%, #f0fdf4 100%);
    background-attachment: fixed;
}
</style>
"""


def inject_page_bg():
    st.markdown(PAGE_BG_CSS, unsafe_allow_html=True)


def render_header_card():
    """Header card: logo left, title + description + tags right. No border, subtle shadow only."""
    logo_path = BASE / "logo.png"
    logo_b64 = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
    img_html = f'<img src="data:image/png;base64,{logo_b64}" style="width:100px;height:100px;object-fit:contain;border-radius:10px;" />' if logo_b64 else ""
    html = f"""
    <div style="
        background:linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 40%, #ecfdf5 100%);
        border-radius:16px;
        border:none;
        padding:1.25rem 1.5rem;
        margin-bottom:1.5rem;
        display:flex;
        align-items:center;
        gap:1.5rem;
    ">
        <div style="flex-shrink:0;">{img_html}</div>
        <div style="flex:1;">
            <div style="font-size:0.7rem;color:#475569;letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.25rem;">UW x DATABRICKS HACKATHON</div>
            <h1 style="font-size:1.75rem;font-weight:700;color:#0f172a;margin:0 0 0.35rem 0;">TerraCast</h1>
            <p style="font-size:0.9rem;color:#475569;margin:0 0 0.75rem 0;line-height:1.5;">Weather-to-yield intelligence for exploring crop patterns, spotting anomalies, and testing prediction scenarios.</p>
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                <span style="background:#6366f1;color:white;padding:0.35rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:500;">NOAA Weather</span>
                <span style="background:#059669;color:white;padding:0.35rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:500;">Yield Signals</span>
                <span style="background:#0ea5e9;color:white;padding:0.35rem 0.75rem;border-radius:999px;font-size:0.8rem;font-weight:500;">ML Prediction</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_report_page():
    """Report page content with Back button."""
    inject_page_bg()
    if st.button("← Back to main", key="back_from_report"):
        st.session_state["page"] = "main"
        st.rerun()
    logo_path = BASE / "logo.png"
    if logo_path.exists():
        col_a, col_logo, col_b = st.columns([1, 2, 1])
        with col_logo:
            st.image(str(logo_path), use_container_width=False, width=160)
    st.title("Report")
    st.caption("Weather–Yield summary and insights")
    st.markdown("---")
    st.markdown("""
    ### Summary
    - **Data**: RMA County Yields + NOAA Weather (merged).
    - **Features**: State/county-level yield, crop (Corn/Soybeans), year, and weather variables (avg temp max/min, precipitation).
    - **Tools**: Map (state/county), anomaly detection, yield prediction (ML), and chat.
    """)
    st.markdown("### Quick stats")
    df = load_data()
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total records", f"{len(df):,}", "rows")
        with c2:
            st.metric("Years", f"{int(df['year'].min())} – {int(df['year'].max())}", "")
        with c3:
            st.metric("States", df["State Abbreviation"].nunique(), "")
    st.markdown("---")
    st.markdown("*Add more report sections (charts, exports, insights) here.*")


def main():
    st.set_page_config(page_title="Weather–Yield Explorer", layout="wide")
    inject_page_bg()

    if st.session_state.get("page", "main") == "report":
        render_report_page()
        return

    # Top-right: Report button
    _, nav_col = st.columns([10, 1])
    with nav_col:
        if st.button("Report", key="go_to_report"):
            st.session_state["page"] = "report"
            st.rerun()

    render_header_card()

    df = load_data()
    crops = sorted(df["Commodity Name"].dropna().unique())
    years = sorted(df["year"].dropna().astype(int).unique())
    weather_map = {
        "Yield (mean bu/acre)": "yield",
        "Avg max temperature (°C)": "avg_temp_max_c",
        "Avg min temperature (°C)": "avg_temp_min_c",
        "Total precipitation (mm)": "total_precipitation_mm",
    }
    model, enc_state, enc_crop = train_yield_model(df)
    df_median = df[["avg_temp_max_c", "avg_temp_min_c", "total_precipitation_mm"]].median().to_dict() if "avg_temp_max_c" in df.columns else {}

    col_left, col_right = st.columns([1, 3])
    with col_left:
        render_chat()

    with col_right:
        # Filters above the map
        st.subheader("Filters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            crop = st.selectbox("Crop", options=crops, index=0 if "Corn" in crops else 0)
        with c2:
            year = st.selectbox("Year", options=years, index=len(years) - 1 if years else 0)
        with c3:
            color_by = st.selectbox(
                "Variable to display",
                options=list(weather_map.keys()),
                index=0,
            )
        with c4:
            map_level = st.radio("Map level", options=["State-level", "County-level"], index=0, horizontal=True)
        color_var = weather_map.get(color_by, "yield")
        selected_states = st.multiselect(
            "States (leave empty for all)",
            options=STATE_ABBR,
            default=[],
            key="states",
        )
        st.divider()

        # Right side: Map + Prediction panel in one row
        map_col, pred_col = st.columns([2, 1])
        with pred_col:
            st.subheader("Yield prediction")
            pred_crop = st.selectbox("Crop (predict)", options=crops, index=0, key="pred_crop")
            pred_state = st.selectbox("State", options=STATE_ABBR, index=0, key="pred_state")
            pred_year = st.number_input("Year", min_value=2010, max_value=2030, value=year if years else 2024, key="pred_year")
            use_weather = st.checkbox("Override weather (optional)", value=False, key="use_weather")
            pred_tmax = pred_tmin = pred_precip = None
            if use_weather:
                pred_tmax = st.number_input("Avg max temp (°C)", value=22.0, format="%.1f", key="ptmax")
                pred_tmin = st.number_input("Avg min temp (°C)", value=8.0, format="%.1f", key="ptmin")
                pred_precip = st.number_input("Total precip (mm)", value=5000.0, format="%.0f", key="pprecip")
            show_pred_on_map = st.checkbox("Show prediction on map", value=False, key="show_pred_map")
            if st.button("Predict yield", key="predict_btn"):
                pred_val = predict_yield(
                    model, enc_state, enc_crop, pred_crop, pred_state, int(pred_year),
                    temp_max=pred_tmax, temp_min=pred_tmin, precip=pred_precip, df_median=df_median,
                )
                st.session_state["last_prediction"] = pred_val
            if "last_prediction" in st.session_state and st.session_state["last_prediction"] is not None:
                st.metric("Predicted yield", f"{st.session_state['last_prediction']:.1f}", "bu/acre")

        # State-level aggregation
        agg = state_yield_agg(df, crop, year, color_var if color_var != "yield" else None)
        if agg.empty:
            st.warning(f"No data for crop={crop}, year={year}.")
            return

        if selected_states:
            agg = agg[agg["State Abbreviation"].isin(selected_states)]
        if agg.empty:
            st.warning("No data for selected states.")
            return

        if color_var == "yield":
            agg["map_value"] = agg["yield_mean"]
            color_label = "Mean yield (bu/acre)"
        else:
            agg["map_value"] = agg.get("weather_value", agg["yield_mean"])
            color_label = color_by

        # Optional: show prediction on map (state-level predicted yield)
        show_pred_on_map = st.session_state.get("show_pred_map", False)
        if show_pred_on_map and model is not None and color_var == "yield":
            pred_values = []
            for _, row in agg.iterrows():
                p = predict_yield(model, enc_state, enc_crop, crop, row["State Abbreviation"], year, df_median=df_median)
                pred_values.append(p if p is not None else row["map_value"])
            agg["map_value"] = pred_values
            color_label = "Predicted yield (bu/acre)"

        with map_col:
            st.subheader("Map")
            show_state = map_level == "State-level"
            if show_state:
                fig = px.choropleth(
                    agg,
                    locations="State Abbreviation",
                    locationmode="USA-states",
                    color="map_value",
                    scope="usa",
                    color_continuous_scale="YlGnBu" if color_var == "yield" else "RdYlBu_r",
                    labels={"map_value": color_label},
                    title=f"{crop} · {year} — {color_label} ({map_level})",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                if "fips" not in df.columns:
                    st.caption("County map requires FIPS (State Code + County Code) in data.")
                else:
                    sub_county = df[(df["Commodity Name"] == crop) & (df["year"] == year)].copy()
                    if selected_states:
                        sub_county = sub_county[sub_county["State Abbreviation"].isin(selected_states)]
                    if sub_county.empty:
                        st.caption("No county data for selected states.")
                    else:
                        if color_var == "yield":
                            sub_county["map_value"] = sub_county["Yield Amount"]
                        else:
                            sub_county["map_value"] = sub_county.get(color_var, sub_county["Yield Amount"])
                        county_agg = sub_county.groupby("fips").agg(
                            map_value=("map_value", "mean"),
                            State_Abbreviation=("State Abbreviation", "first"),
                        ).reset_index()
                        county_agg = county_agg[county_agg["map_value"].notna()]
                        if county_agg.empty:
                            st.caption("No county-level values to plot for this selection.")
                        else:
                            if show_pred_on_map and model is not None:
                                for i, row in county_agg.iterrows():
                                    st_abbr = row["State_Abbreviation"]
                                    p = predict_yield(model, enc_state, enc_crop, crop, st_abbr, year, df_median=df_median)
                                    if p is not None:
                                        county_agg.at[i, "map_value"] = p
                            try:
                                geojson = load_county_geojson()
                                fig_county = go.Figure(
                                    go.Choropleth(
                                        geojson=geojson,
                                        locations=county_agg["fips"],
                                        z=county_agg["map_value"],
                                        featureidkey="id",
                                        colorscale="YlGnBu" if color_var == "yield" else "RdYlBu_r",
                                        colorbar_title=color_label,
                                        showscale=True,
                                    )
                                )
                                fig_county.update_layout(
                                    title=f"{crop} · {year} — {color_label} ({map_level})",
                                    geo=dict(scope="usa", fitbounds="locations"),
                                    margin=dict(l=0, r=0, t=40, b=0),
                                    height=500,
                                )
                                st.plotly_chart(fig_county, use_container_width=True)
                            except Exception as e:
                                st.caption(f"County map skipped: {e}")

        # State summary table
        st.subheader("State summary")
        display = agg.rename(columns={
            "State Abbreviation": "State",
            "yield_mean": "Mean yield",
            "yield_sum": "Yield sum",
            "county_count": "Counties",
        })
        cols = [c for c in ["State", "Mean yield", "Yield sum", "Counties"] if c in display.columns]
        if color_var != "yield" and "weather_value" in agg.columns and agg["weather_value"].notna().any():
            display["Weather value"] = agg["weather_value"].round(2)
            cols.append("Weather value")
        st.dataframe(display[cols].sort_values("Mean yield", ascending=False), use_container_width=True)

        # ---------- Anomaly Detection ----------
        st.divider()
        st.subheader("Anomaly Detection")
        st.caption("Flag counties with unusually high or low yield (Z-score) or yield–weather mismatch (residual).")

        with st.expander("Anomaly settings", expanded=True):
            method = st.radio(
                "Method",
                options=[
                    "Yield Z-score (by crop & year)",
                    "Yield vs weather residual (regression)",
                ],
                index=0,
                help="Z-score: flag counties far from that year's mean. Residual: flag where yield doesn't match weather-based expectation.",
            )
            threshold = st.slider("Threshold (standard deviations)", min_value=1.0, max_value=3.0, value=2.0, step=0.25)

        if method == "Yield Z-score (by crop & year)":
            anomalies = detect_yield_zscore(df, crop, year, threshold_std=threshold)
        else:
            anomalies = detect_yield_weather_residual(df, crop, year, threshold_std=threshold)

        if anomalies.empty:
            st.info("No anomalies found for this crop/year with the chosen threshold. Try lowering the threshold or another method.")
        else:
            high_n = (anomalies["anomaly_type"].str.contains("High", na=False)).sum()
            low_n = (anomalies["anomaly_type"].str.contains("Low", na=False)).sum()
            st.metric("Flagged counties", len(anomalies), f"High: {high_n} · Low: {low_n}")

            show_cols = ["State Abbreviation", "State Name", "County Name", "year", "Yield Amount", "anomaly_type"]
            if "yield_zscore" in anomalies.columns:
                anomalies["yield_zscore"] = anomalies["yield_zscore"].round(3)
                show_cols.append("yield_zscore")
            if "yield_predicted" in anomalies.columns:
                anomalies["yield_predicted"] = anomalies["yield_predicted"].round(2)
                anomalies["residual"] = anomalies["residual"].round(2)
                show_cols.extend(["yield_predicted", "residual"])
            table_df = anomalies[[c for c in show_cols if c in anomalies.columns]].rename(columns={
                "State Abbreviation": "State",
                "State Name": "State name",
                "County Name": "County",
                "year": "Year",
                "Yield Amount": "Yield",
                "anomaly_type": "Type",
                "yield_zscore": "Z-score",
            })
            st.dataframe(table_df, use_container_width=True)

            state_anom = anomalies.groupby("State Abbreviation").size().reset_index(name="anomaly_count")
            fig_anom = px.choropleth(
                state_anom,
                locations="State Abbreviation",
                locationmode="USA-states",
                color="anomaly_count",
                scope="usa",
                color_continuous_scale="Reds",
                labels={"anomaly_count": "Flagged counties"},
                title=f"Anomaly count by state — {crop} {year}",
            )
            fig_anom.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=400)
            st.plotly_chart(fig_anom, use_container_width=True)

        with st.expander("What-Drives-Yield (non-technical summary)"):
            st.markdown("""
            - **Map color**: Darker shades mean higher values for the selected variable (yield or weather) in that state.
            - **Crop and year**: Switch between Corn and Soybeans and change the year to compare across time.
            - **Weather and yield**: Choosing temperature or precipitation shows how that variable varies by state for the selected year.  
              Correlation and anomaly analysis can be added later and shown on the map.
            """)


if __name__ == "__main__":
    main()
