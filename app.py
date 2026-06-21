"""
Local Food Wastage Management System
--------------------------------------
A Streamlit application that connects surplus food providers with receivers,
stores data in SQLite, supports full CRUD, runs 15 analytical SQL queries,
and renders a fully dynamic (filter-driven, interactive) EDA dashboard.

Run with:  streamlit run app.py
Requires:  streamlit, pandas, plotly  ->  pip install streamlit pandas plotly
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Local Food Wastage Management System",
    page_icon="🍽️",
    layout="wide",
)

# Build the DB path relative to THIS file, so it works no matter what
# directory streamlit is launched from (fixes the "unable to open database
# file" error that comes from using a relative "../database/..." path).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "food_wastage.db")


# --------------------------------------------------------------------------
# DB HELPERS
# --------------------------------------------------------------------------
def get_connection():
    if not os.path.exists(DB_PATH):
        st.error(
            f"Database not found at `{DB_PATH}`. "
            "Make sure food_wastage.db exists inside a `database/` folder "
            "next to app.py."
        )
        st.stop()
    return sqlite3.connect(DB_PATH)


def run_query(query: str, params=None) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()
    return df


def run_action(query: str, params=None):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params or ())
        conn.commit()
    finally:
        conn.close()


@st.cache_data(ttl=15)
def load_tables():
    conn = get_connection()
    try:
        providers = pd.read_sql("SELECT * FROM providers", conn)
        receivers = pd.read_sql("SELECT * FROM receivers", conn)
        food = pd.read_sql("SELECT * FROM food_listings", conn)
        claims = pd.read_sql("SELECT * FROM claims", conn)
    finally:
        conn.close()
    return providers, receivers, food, claims


def refresh_data():
    """Clear cache after any write so every page sees fresh data."""
    load_tables.clear()
    st.cache_data.clear()


@st.cache_data(ttl=15)
def build_merged(providers: pd.DataFrame, receivers: pd.DataFrame,
                  food: pd.DataFrame, claims: pd.DataFrame):
    """Denormalised, analysis-ready views used across the app."""
    prov = providers.rename(columns={
        "Name": "Provider_Name",
        "Type": "Provider_Type_Master",
        "Address": "Provider_Address",
        "City": "Provider_City",
        "Contact": "Provider_Contact",
    })
    recv = receivers.rename(columns={
        "Name": "Receiver_Name",
        "Type": "Receiver_Type",
        "City": "Receiver_City",
        "Contact": "Receiver_Contact",
    })

    food_full = food.merge(prov, on="Provider_ID", how="left")
    if "Expiry_Date" in food_full.columns:
        food_full["Expiry_Date_dt"] = pd.to_datetime(
            food_full["Expiry_Date"], errors="coerce"
        )

    claims_full = (
        claims.merge(food_full, on="Food_ID", how="left")
        .merge(recv, on="Receiver_ID", how="left")
    )
    if "Timestamp" in claims_full.columns:
        claims_full["Timestamp_dt"] = pd.to_datetime(
            claims_full["Timestamp"], errors="coerce"
        )

    return food_full, claims_full


def distinct_values(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return []
    return sorted([v for v in df[col].dropna().unique().tolist()])


# --------------------------------------------------------------------------
# THE 15 SQL QUERIES (kept identical to the ones validated in DB Browser)
# --------------------------------------------------------------------------
QUERIES = [
    {
        "title": "1. How many providers in each city?",
        "sql": """SELECT City, COUNT(*) AS Provider_Count
FROM providers
GROUP BY City
ORDER BY Provider_Count DESC;""",
    },
    {
        "title": "2. How many receivers in each city?",
        "sql": """SELECT City, COUNT(*) AS Receiver_Count
FROM receivers
GROUP BY City
ORDER BY Receiver_Count DESC;""",
    },
    {
        "title": "3. Which provider type contributes the most food?",
        "sql": """SELECT Provider_Type, SUM(Quantity) AS Total_Food
FROM food_listings
GROUP BY Provider_Type
ORDER BY Total_Food DESC;""",
    },
    {
        "title": "4. Contact details of providers in a specific city",
        "sql": """SELECT Name, Contact, City
FROM providers
WHERE City = ?;""",
        "param": "city",
    },
    {
        "title": "5. Which receivers have claimed the most food?",
        "sql": """SELECT r.Name, COUNT(*) AS Total_Claims
FROM claims c
JOIN receivers r ON c.Receiver_ID = r.Receiver_ID
GROUP BY r.Name
ORDER BY Total_Claims DESC;""",
    },
    {
        "title": "6. What is the total quantity of food available?",
        "sql": """SELECT SUM(Quantity) AS Total_Food_Available
FROM food_listings;""",
    },
    {
        "title": "7. Which city has the highest number of food listings?",
        "sql": """SELECT Location, COUNT(*) AS Listings
FROM food_listings
GROUP BY Location
ORDER BY Listings DESC;""",
    },
    {
        "title": "8. What are the most commonly available food types?",
        "sql": """SELECT Food_Type, COUNT(*) AS Count
FROM food_listings
GROUP BY Food_Type
ORDER BY Count DESC;""",
    },
    {
        "title": "9. How many claims have been made for each food item?",
        "sql": """SELECT f.Food_Name, COUNT(*) AS Claims
FROM claims c
JOIN food_listings f ON c.Food_ID = f.Food_ID
GROUP BY f.Food_Name
ORDER BY Claims DESC;""",
    },
    {
        "title": "10. Which provider has the highest number of successful claims?",
        "sql": """SELECT p.Name, COUNT(*) AS Successful_Claims
FROM claims c
JOIN food_listings f ON c.Food_ID = f.Food_ID
JOIN providers p ON f.Provider_ID = p.Provider_ID
WHERE c.Status = 'Completed'
GROUP BY p.Name
ORDER BY Successful_Claims DESC;""",
    },
    {
        "title": "11. What % of claims are Completed vs Pending vs Cancelled?",
        "sql": """SELECT
    Status,
    COUNT(*) AS Total,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM claims), 2) AS Percentage
FROM claims
GROUP BY Status;""",
    },
    {
        "title": "12. What is the average quantity of food claimed per receiver?",
        "sql": """SELECT r.Name, AVG(f.Quantity) AS Avg_Quantity
FROM claims c
JOIN food_listings f ON c.Food_ID = f.Food_ID
JOIN receivers r ON c.Receiver_ID = r.Receiver_ID
GROUP BY r.Name
ORDER BY Avg_Quantity DESC;""",
    },
    {
        "title": "13. Which meal type is claimed the most?",
        "sql": """SELECT f.Meal_Type, COUNT(*) AS Total_Claims
FROM claims c
JOIN food_listings f ON c.Food_ID = f.Food_ID
GROUP BY f.Meal_Type
ORDER BY Total_Claims DESC;""",
    },
    {
        "title": "14. What is the total quantity of food donated by each provider?",
        "sql": """SELECT p.Name, SUM(f.Quantity) AS Total_Donated
FROM providers p
JOIN food_listings f ON p.Provider_ID = f.Provider_ID
GROUP BY p.Name
ORDER BY Total_Donated DESC;""",
    },
    {
        "title": "15. Which food items are expiring soonest?",
        "sql": """SELECT Food_Name, Expiry_Date, Quantity
FROM food_listings
ORDER BY Expiry_Date ASC
LIMIT 10;""",
    },
]


# --------------------------------------------------------------------------
# PAGE: HOME
# --------------------------------------------------------------------------
def page_home():
    st.title("🍽️ Local Food Wastage Management System")
    st.caption("Connecting surplus food providers with people and organizations who need it.")

    providers, receivers, food, claims = load_tables()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Providers", f"{len(providers):,}")
    c2.metric("Receivers", f"{len(receivers):,}")
    c3.metric("Food Listings", f"{len(food):,}")
    c4.metric("Total Claims", f"{len(claims):,}")

    c5, c6, c7 = st.columns(3)
    total_qty = int(food["Quantity"].sum()) if "Quantity" in food.columns else 0
    completed = int((claims["Status"] == "Completed").sum()) if "Status" in claims.columns else 0
    pending = int((claims["Status"] == "Pending").sum()) if "Status" in claims.columns else 0
    c5.metric("Total Quantity Listed", f"{total_qty:,}")
    c6.metric("Completed Claims", f"{completed:,}")
    c7.metric("Pending Claims", f"{pending:,}")

    st.divider()
    st.markdown(
        """
        ### What you can do here
        - **Browse & Claim Food** — filter live listings by city, provider, food type, and meal type.
        - **Contact Directory** — look up provider / receiver contact details directly.
        - **SQL Queries & Insights** — run the 15 analytical queries used to evaluate this project.
        - **Dynamic EDA Dashboard** — fully interactive charts that update as you change filters.
        - **Manage Data** — add, update, or delete providers, receivers, food listings, and claims.
        """
    )


# --------------------------------------------------------------------------
# PAGE: BROWSE & CLAIM FOOD
# --------------------------------------------------------------------------
def page_browse():
    st.title("🍱 Browse & Claim Food")

    providers, receivers, food, claims = load_tables()
    food_full, claims_full = build_merged(providers, receivers, food, claims)

    st.subheader("Filter listings")
    f1, f2, f3, f4 = st.columns(4)
    city = f1.selectbox("City", ["All"] + distinct_values(food_full, "Location"))
    provider_name = f2.selectbox("Provider", ["All"] + distinct_values(food_full, "Provider_Name"))
    food_type = f3.selectbox("Food Type", ["All"] + distinct_values(food_full, "Food_Type"))
    meal_type = f4.selectbox("Meal Type", ["All"] + distinct_values(food_full, "Meal_Type"))

    filtered = food_full.copy()
    if city != "All":
        filtered = filtered[filtered["Location"] == city]
    if provider_name != "All":
        filtered = filtered[filtered["Provider_Name"] == provider_name]
    if food_type != "All":
        filtered = filtered[filtered["Food_Type"] == food_type]
    if meal_type != "All":
        filtered = filtered[filtered["Meal_Type"] == meal_type]

    st.write(f"**{len(filtered)}** matching listing(s)")
    show_cols = [
        "Food_ID", "Food_Name", "Quantity", "Expiry_Date", "Food_Type",
        "Meal_Type", "Location", "Provider_Name", "Provider_Contact",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(filtered[show_cols], width="stretch", hide_index=True)

    st.divider()
    st.subheader("📞 Need to coordinate pickup? Provider contacts for this view")
    contacts = (
        filtered[["Provider_Name", "Provider_Contact", "Location"]]
        .drop_duplicates()
        .rename(columns={"Provider_Name": "Name", "Provider_Contact": "Contact", "Location": "City"})
    )
    st.dataframe(contacts, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# PAGE: CONTACT DIRECTORY
# --------------------------------------------------------------------------
def page_contacts():
    st.title("📞 Contact Directory")
    providers, receivers, food, claims = load_tables()

    tab1, tab2 = st.tabs(["Providers", "Receivers"])

    with tab1:
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", ["All"] + distinct_values(providers, "City"), key="pc_city")
        ptype = c2.selectbox("Type", ["All"] + distinct_values(providers, "Type"), key="pc_type")
        df = providers.copy()
        if city != "All":
            df = df[df["City"] == city]
        if ptype != "All":
            df = df[df["Type"] == ptype]
        st.dataframe(df, width="stretch", hide_index=True)

    with tab2:
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", ["All"] + distinct_values(receivers, "City"), key="rc_city")
        rtype = c2.selectbox("Type", ["All"] + distinct_values(receivers, "Type"), key="rc_type")
        df = receivers.copy()
        if city != "All":
            df = df[df["City"] == city]
        if rtype != "All":
            df = df[df["Type"] == rtype]
        st.dataframe(df, width="stretch", hide_index=True)


# --------------------------------------------------------------------------
# PAGE: SQL QUERIES & INSIGHTS
# --------------------------------------------------------------------------
def page_queries():
    st.title("📊 SQL Queries & Insights")
    st.caption("All 15 evaluation queries — pick one to run it live against the database.")

    providers, _, _, _ = load_tables()
    titles = [q["title"] for q in QUERIES]
    choice = st.selectbox("Choose a query", titles)
    query_def = next(q for q in QUERIES if q["title"] == choice)

    params = None
    if query_def.get("param") == "city":
        city = st.selectbox("City", distinct_values(providers, "City"))
        params = (city,)

    with st.expander("Show SQL"):
        st.code(query_def["sql"], language="sql")

    try:
        result = run_query(query_def["sql"], params=params)
        st.dataframe(result, width="stretch", hide_index=True)

        # Light auto-chart for two-column numeric results
        if result.shape[1] == 2 and pd.api.types.is_numeric_dtype(result.iloc[:, 1]):
            fig = px.bar(result, x=result.columns[0], y=result.columns[1])
            st.plotly_chart(fig, width="stretch")
    except Exception as e:
        st.error(f"Query failed: {e}")

    st.divider()
    if st.button("▶️ Run all 15 queries"):
        for q in QUERIES:
            st.markdown(f"**{q['title']}**")
            try:
                if q.get("param") == "city":
                    sample_city = distinct_values(providers, "City")
                    p = (sample_city[0],) if sample_city else None
                else:
                    p = None
                df = run_query(q["sql"], params=p)
                st.dataframe(df, width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Failed: {e}")


# --------------------------------------------------------------------------
# PAGE: DYNAMIC EDA DASHBOARD
# --------------------------------------------------------------------------
def page_eda():
    st.title("📈 Dynamic EDA Dashboard")
    st.caption("Every chart below recalculates live from the filters you choose.")

    providers, receivers, food, claims = load_tables()
    food_full, claims_full = build_merged(providers, receivers, food, claims)

    # ---------------- Filter panel ----------------
    st.subheader("Filters")
    f1, f2, f3, f4, f5 = st.columns(5)
    city_sel = f1.multiselect("City", distinct_values(food_full, "Location"))
    ptype_sel = f2.multiselect("Provider Type", distinct_values(food_full, "Provider_Type"))
    ftype_sel = f3.multiselect("Food Type", distinct_values(food_full, "Food_Type"))
    mtype_sel = f4.multiselect("Meal Type", distinct_values(food_full, "Meal_Type"))
    status_sel = f5.multiselect("Claim Status", distinct_values(claims_full, "Status"))

    top_n = st.slider("Top N (for ranking charts)", min_value=3, max_value=20, value=10)

    def apply_food_filters(df):
        out = df.copy()
        if city_sel:
            out = out[out["Location"].isin(city_sel)]
        if ptype_sel:
            out = out[out["Provider_Type"].isin(ptype_sel)]
        if ftype_sel:
            out = out[out["Food_Type"].isin(ftype_sel)]
        if mtype_sel:
            out = out[out["Meal_Type"].isin(mtype_sel)]
        return out

    def apply_claims_filters(df):
        out = apply_food_filters(df)
        if status_sel:
            out = out[out["Status"].isin(status_sel)]
        return out

    f_food = apply_food_filters(food_full)
    f_claims = apply_claims_filters(claims_full)

    # Providers / receivers filtered only by city (no food-specific columns there)
    f_providers = providers[providers["City"].isin(city_sel)] if city_sel else providers
    f_receivers = receivers[receivers["City"].isin(city_sel)] if city_sel else receivers

    st.info(f"Showing **{len(f_food)}** food listings and **{len(f_claims)}** claims for the current filters.")
    st.divider()

    # ---------------- Row 1: Provider / Receiver type ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Provider Type Distribution**")
        d = f_providers["Type"].value_counts().reset_index()
        d.columns = ["Provider Type", "Count"]
        fig = px.bar(d, x="Provider Type", y="Count", color="Provider Type", text="Count")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown("**Receiver Type Distribution**")
        d = f_receivers["Type"].value_counts().reset_index()
        d.columns = ["Receiver Type", "Count"]
        fig = px.bar(d, x="Receiver Type", y="Count", color="Receiver Type", text="Count")
        st.plotly_chart(fig, width="stretch")

    # ---------------- Row 2: Food type / Meal type ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Food Type Distribution**")
        d = f_food["Food_Type"].value_counts().reset_index()
        d.columns = ["Food Type", "Count"]
        fig = px.pie(d, names="Food Type", values="Count", hole=0.35)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown("**Meal Type Distribution**")
        d = f_food["Meal_Type"].value_counts().reset_index()
        d.columns = ["Meal Type", "Count"]
        fig = px.bar(d, x="Meal Type", y="Count", color="Meal Type", text="Count")
        st.plotly_chart(fig, width="stretch")

    # ---------------- Row 3: Claim status / Quantity distribution ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Claim Status Distribution**")
        if len(f_claims):
            d = f_claims["Status"].value_counts().reset_index()
            d.columns = ["Status", "Count"]
            fig = px.pie(d, names="Status", values="Count", hole=0.35)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No claims match the current filters.")
    with col2:
        st.markdown("**Food Quantity Distribution**")
        if len(f_food):
            qty_min = float(f_food["Quantity"].min())
            qty_max = float(f_food["Quantity"].max())
            bin_size = max(1, (qty_max - qty_min) / 15) if qty_max > qty_min else 1
            fig = px.histogram(f_food, x="Quantity")
            fig.update_traces(
                xbins=dict(start=qty_min, end=qty_max + bin_size, size=bin_size),
                marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
            )
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No food listings match the current filters.")

    st.divider()

    # ---------------- Row 4: Top cities ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top {top_n} Cities by Providers**")
        d = f_providers["City"].value_counts().head(top_n).reset_index()
        d.columns = ["City", "Providers"]
        fig = px.bar(d, x="City", y="Providers", text="Providers")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown(f"**Top {top_n} Cities by Receivers**")
        d = f_receivers["City"].value_counts().head(top_n).reset_index()
        d.columns = ["City", "Receivers"]
        fig = px.bar(d, x="City", y="Receivers", text="Receivers")
        st.plotly_chart(fig, width="stretch")

    # ---------------- Row 5: Top providers / Top claimed foods ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top {top_n} Providers by Total Donation**")
        if len(f_food):
            d = (
                f_food.groupby("Provider_Name")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )
            fig = px.bar(d, x="Provider_Name", y="Quantity")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No data for current filters.")
    with col2:
        st.markdown(f"**Top {top_n} Most Claimed Food Items**")
        if len(f_claims):
            d = (
                f_claims.groupby("Food_Name")
                .size()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index(name="Claims")
            )
            fig = px.bar(d, x="Food_Name", y="Claims")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("No claims for current filters.")

    st.divider()

    # ---------------- Row 6: Claims trend over time ----------------
    st.markdown("**Claims Trend Over Time**")
    if len(f_claims) and "Timestamp_dt" in f_claims.columns and f_claims["Timestamp_dt"].notna().any():
        trend = (
            f_claims.dropna(subset=["Timestamp_dt"])
            .assign(Day=lambda d: d["Timestamp_dt"].dt.date)
            .groupby("Day")
            .size()
            .reset_index(name="Claims")
        )
        fig = px.line(trend, x="Day", y="Claims", markers=True)
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No timestamped claims available for the current filters.")

    # ---------------- Row 7: Food expiring soon ----------------
    st.markdown("**Food Expiring Soonest**")
    if len(f_food) and "Expiry_Date_dt" in f_food.columns:
        soon = f_food.dropna(subset=["Expiry_Date_dt"]).sort_values("Expiry_Date_dt").head(top_n)
        if len(soon):
            fig = px.bar(
                soon, x="Food_Name", y="Quantity",
                color=soon["Expiry_Date_dt"].astype(str),
                labels={"color": "Expiry Date"},
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(
                soon[["Food_Name", "Quantity", "Expiry_Date", "Location", "Provider_Name"]],
                width="stretch", hide_index=True,
            )
        else:
            st.warning("No upcoming expiry dates in the current filter.")
    else:
        st.warning("No food listings match the current filters.")


# --------------------------------------------------------------------------
# PAGE: MANAGE DATA (CRUD)
# --------------------------------------------------------------------------
def crud_providers():
    providers, _, _, _ = load_tables()
    action = st.radio("Action", ["View All", "Add", "Update", "Delete"], horizontal=True, key="prov_action")

    if action == "View All":
        st.dataframe(providers, width="stretch", hide_index=True)

    elif action == "Add":
        with st.form("add_provider"):
            name = st.text_input("Name")
            type_options = distinct_values(providers, "Type") or ["Restaurant", "Grocery Store", "Supermarket"]
            ptype = st.selectbox("Type", type_options + ["Other (type below)"])
            if ptype == "Other (type below)":
                ptype = st.text_input("Custom Type")
            address = st.text_input("Address")
            city = st.text_input("City")
            contact = st.text_input("Contact")
            submitted = st.form_submit_button("Add Provider")
            if submitted:
                new_id = int(providers["Provider_ID"].max()) + 1 if len(providers) else 1
                run_action(
                    "INSERT INTO providers (Provider_ID, Name, Type, Address, City, Contact) VALUES (?,?,?,?,?,?)",
                    (new_id, name, ptype, address, city, contact),
                )
                refresh_data()
                st.success(f"Provider '{name}' added with ID {new_id}.")
                st.rerun()

    elif action == "Update":
        if not len(providers):
            st.warning("No providers to update.")
            return
        pid = st.selectbox("Select Provider_ID", providers["Provider_ID"].tolist())
        row = providers[providers["Provider_ID"] == pid].iloc[0]
        with st.form("update_provider"):
            name = st.text_input("Name", row["Name"])
            ptype = st.text_input("Type", row["Type"])
            address = st.text_input("Address", row["Address"])
            city = st.text_input("City", row["City"])
            contact = st.text_input("Contact", row["Contact"])
            submitted = st.form_submit_button("Update Provider")
            if submitted:
                run_action(
                    "UPDATE providers SET Name=?, Type=?, Address=?, City=?, Contact=? WHERE Provider_ID=?",
                    (name, ptype, address, city, contact, int(pid)),
                )
                refresh_data()
                st.success(f"Provider {pid} updated.")
                st.rerun()

    elif action == "Delete":
        if not len(providers):
            st.warning("No providers to delete.")
            return
        pid = st.selectbox("Select Provider_ID to delete", providers["Provider_ID"].tolist())
        st.dataframe(providers[providers["Provider_ID"] == pid], width="stretch", hide_index=True)
        confirm = st.checkbox("I confirm I want to delete this provider.")
        if st.button("Delete Provider", disabled=not confirm):
            run_action("DELETE FROM providers WHERE Provider_ID=?", (int(pid),))
            refresh_data()
            st.success(f"Provider {pid} deleted.")
            st.rerun()


def crud_receivers():
    _, receivers, _, _ = load_tables()
    action = st.radio("Action", ["View All", "Add", "Update", "Delete"], horizontal=True, key="recv_action")

    if action == "View All":
        st.dataframe(receivers, width="stretch", hide_index=True)

    elif action == "Add":
        with st.form("add_receiver"):
            name = st.text_input("Name")
            type_options = distinct_values(receivers, "Type") or ["NGO", "Community Center", "Individual"]
            rtype = st.selectbox("Type", type_options + ["Other (type below)"])
            if rtype == "Other (type below)":
                rtype = st.text_input("Custom Type")
            city = st.text_input("City")
            contact = st.text_input("Contact")
            submitted = st.form_submit_button("Add Receiver")
            if submitted:
                new_id = int(receivers["Receiver_ID"].max()) + 1 if len(receivers) else 1
                run_action(
                    "INSERT INTO receivers (Receiver_ID, Name, Type, City, Contact) VALUES (?,?,?,?,?)",
                    (new_id, name, rtype, city, contact),
                )
                refresh_data()
                st.success(f"Receiver '{name}' added with ID {new_id}.")
                st.rerun()

    elif action == "Update":
        if not len(receivers):
            st.warning("No receivers to update.")
            return
        rid = st.selectbox("Select Receiver_ID", receivers["Receiver_ID"].tolist())
        row = receivers[receivers["Receiver_ID"] == rid].iloc[0]
        with st.form("update_receiver"):
            name = st.text_input("Name", row["Name"])
            rtype = st.text_input("Type", row["Type"])
            city = st.text_input("City", row["City"])
            contact = st.text_input("Contact", row["Contact"])
            submitted = st.form_submit_button("Update Receiver")
            if submitted:
                run_action(
                    "UPDATE receivers SET Name=?, Type=?, City=?, Contact=? WHERE Receiver_ID=?",
                    (name, rtype, city, contact, int(rid)),
                )
                refresh_data()
                st.success(f"Receiver {rid} updated.")
                st.rerun()

    elif action == "Delete":
        if not len(receivers):
            st.warning("No receivers to delete.")
            return
        rid = st.selectbox("Select Receiver_ID to delete", receivers["Receiver_ID"].tolist())
        st.dataframe(receivers[receivers["Receiver_ID"] == rid], width="stretch", hide_index=True)
        confirm = st.checkbox("I confirm I want to delete this receiver.")
        if st.button("Delete Receiver", disabled=not confirm):
            run_action("DELETE FROM receivers WHERE Receiver_ID=?", (int(rid),))
            refresh_data()
            st.success(f"Receiver {rid} deleted.")
            st.rerun()


def crud_food():
    providers, _, food, _ = load_tables()
    action = st.radio("Action", ["View All", "Add", "Update", "Delete"], horizontal=True, key="food_action")

    if action == "View All":
        st.dataframe(food, width="stretch", hide_index=True)

    elif action == "Add":
        if not len(providers):
            st.warning("Add at least one provider first.")
            return
        with st.form("add_food"):
            food_name = st.text_input("Food Name")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            expiry = st.date_input("Expiry Date")
            provider_label = st.selectbox(
                "Provider",
                [f"{r.Provider_ID} - {r.Name}" for r in providers.itertuples()],
            )
            provider_id = int(provider_label.split(" - ")[0])
            provider_row = providers[providers["Provider_ID"] == provider_id].iloc[0]
            provider_type = st.text_input("Provider Type", provider_row["Type"])
            location = st.text_input("Location (city)", provider_row["City"])
            ftype_options = distinct_values(food, "Food_Type") or ["Vegetarian", "Non-Vegetarian", "Vegan"]
            food_type = st.selectbox("Food Type", ftype_options + ["Other (type below)"])
            if food_type == "Other (type below)":
                food_type = st.text_input("Custom Food Type")
            mtype_options = distinct_values(food, "Meal_Type") or ["Breakfast", "Lunch", "Dinner", "Snacks"]
            meal_type = st.selectbox("Meal Type", mtype_options + ["Other (type below)"])
            if meal_type == "Other (type below)":
                meal_type = st.text_input("Custom Meal Type")
            submitted = st.form_submit_button("Add Food Listing")
            if submitted:
                new_id = int(food["Food_ID"].max()) + 1 if len(food) else 1
                run_action(
                    """INSERT INTO food_listings
                       (Food_ID, Food_Name, Quantity, Expiry_Date, Provider_ID,
                        Provider_Type, Location, Food_Type, Meal_Type)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (new_id, food_name, int(quantity), expiry.isoformat(), provider_id,
                     provider_type, location, food_type, meal_type),
                )
                refresh_data()
                st.success(f"Food listing '{food_name}' added with ID {new_id}.")
                st.rerun()

    elif action == "Update":
        if not len(food):
            st.warning("No food listings to update.")
            return
        fid = st.selectbox("Select Food_ID", food["Food_ID"].tolist())
        row = food[food["Food_ID"] == fid].iloc[0]
        with st.form("update_food"):
            food_name = st.text_input("Food Name", row["Food_Name"])
            quantity = st.number_input("Quantity", min_value=0, value=int(row["Quantity"]), step=1)
            try:
                default_expiry = pd.to_datetime(row["Expiry_Date"]).date()
            except Exception:
                default_expiry = datetime.now().date()
            expiry = st.date_input("Expiry Date", default_expiry)
            provider_type = st.text_input("Provider Type", row["Provider_Type"])
            location = st.text_input("Location", row["Location"])
            food_type = st.text_input("Food Type", row["Food_Type"])
            meal_type = st.text_input("Meal Type", row["Meal_Type"])
            submitted = st.form_submit_button("Update Food Listing")
            if submitted:
                run_action(
                    """UPDATE food_listings
                       SET Food_Name=?, Quantity=?, Expiry_Date=?, Provider_Type=?,
                           Location=?, Food_Type=?, Meal_Type=?
                       WHERE Food_ID=?""",
                    (food_name, int(quantity), expiry.isoformat(), provider_type,
                     location, food_type, meal_type, int(fid)),
                )
                refresh_data()
                st.success(f"Food listing {fid} updated.")
                st.rerun()

    elif action == "Delete":
        if not len(food):
            st.warning("No food listings to delete.")
            return
        fid = st.selectbox("Select Food_ID to delete", food["Food_ID"].tolist())
        st.dataframe(food[food["Food_ID"] == fid], width="stretch", hide_index=True)
        confirm = st.checkbox("I confirm I want to delete this food listing.")
        if st.button("Delete Food Listing", disabled=not confirm):
            run_action("DELETE FROM food_listings WHERE Food_ID=?", (int(fid),))
            refresh_data()
            st.success(f"Food listing {fid} deleted.")
            st.rerun()


def crud_claims():
    _, receivers, food, claims = load_tables()
    action = st.radio("Action", ["View All", "Add", "Update", "Delete"], horizontal=True, key="claim_action")

    if action == "View All":
        st.dataframe(claims, width="stretch", hide_index=True)

    elif action == "Add":
        if not len(food) or not len(receivers):
            st.warning("Add at least one food listing and one receiver first.")
            return
        with st.form("add_claim"):
            food_label = st.selectbox(
                "Food Item",
                [f"{r.Food_ID} - {r.Food_Name}" for r in food.itertuples()],
            )
            food_id = int(food_label.split(" - ")[0])
            receiver_label = st.selectbox(
                "Receiver",
                [f"{r.Receiver_ID} - {r.Name}" for r in receivers.itertuples()],
            )
            receiver_id = int(receiver_label.split(" - ")[0])
            status_options = distinct_values(claims, "Status") or ["Pending", "Completed", "Cancelled"]
            status = st.selectbox("Status", status_options)
            timestamp = st.text_input(
                "Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            submitted = st.form_submit_button("Add Claim")
            if submitted:
                new_id = int(claims["Claim_ID"].max()) + 1 if len(claims) else 1
                run_action(
                    "INSERT INTO claims (Claim_ID, Food_ID, Receiver_ID, Status, Timestamp) VALUES (?,?,?,?,?)",
                    (new_id, food_id, receiver_id, status, timestamp),
                )
                refresh_data()
                st.success(f"Claim added with ID {new_id}.")
                st.rerun()

    elif action == "Update":
        if not len(claims):
            st.warning("No claims to update.")
            return
        cid = st.selectbox("Select Claim_ID", claims["Claim_ID"].tolist())
        row = claims[claims["Claim_ID"] == cid].iloc[0]
        with st.form("update_claim"):
            status_options = distinct_values(claims, "Status") or ["Pending", "Completed", "Cancelled"]
            current = row["Status"] if row["Status"] in status_options else status_options[0]
            status = st.selectbox("Status", status_options, index=status_options.index(current))
            timestamp = st.text_input("Timestamp", row["Timestamp"])
            submitted = st.form_submit_button("Update Claim")
            if submitted:
                run_action(
                    "UPDATE claims SET Status=?, Timestamp=? WHERE Claim_ID=?",
                    (status, timestamp, int(cid)),
                )
                refresh_data()
                st.success(f"Claim {cid} updated.")
                st.rerun()

    elif action == "Delete":
        if not len(claims):
            st.warning("No claims to delete.")
            return
        cid = st.selectbox("Select Claim_ID to delete", claims["Claim_ID"].tolist())
        st.dataframe(claims[claims["Claim_ID"] == cid], width="stretch", hide_index=True)
        confirm = st.checkbox("I confirm I want to delete this claim.")
        if st.button("Delete Claim", disabled=not confirm):
            run_action("DELETE FROM claims WHERE Claim_ID=?", (int(cid),))
            refresh_data()
            st.success(f"Claim {cid} deleted.")
            st.rerun()


def page_manage():
    st.title("🛠️ Manage Data")
    st.caption("Full CRUD — add, update, or delete records in any table.")
    tab1, tab2, tab3, tab4 = st.tabs(["Providers", "Receivers", "Food Listings", "Claims"])
    with tab1:
        crud_providers()
    with tab2:
        crud_receivers()
    with tab3:
        crud_food()
    with tab4:
        crud_claims()


# --------------------------------------------------------------------------
# MAIN / NAVIGATION
# --------------------------------------------------------------------------
def main():
    st.sidebar.title("🍽️ Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Browse & Claim Food",
            "Contact Directory",
            "SQL Queries & Insights",
            "Dynamic EDA Dashboard",
            "Manage Data",
        ],
    )

    if page == "Home":
        page_home()
    elif page == "Browse & Claim Food":
        page_browse()
    elif page == "Contact Directory":
        page_contacts()
    elif page == "SQL Queries & Insights":
        page_queries()
    elif page == "Dynamic EDA Dashboard":
        page_eda()
    elif page == "Manage Data":
        page_manage()


if __name__ == "__main__":
    main()