import pandas as pd
import streamlit as st
import os
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('food_pantry_dataset.csv')

# Step 2: Prepare features and labels
X = data['Food'].str.lower() 
y = data['Perishable']

# Step 3: Convert text to numeric features
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Step 5: Train Logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Predict on test set
y_pred = model.predict(X_test)
st.sidebar.write("Model Performance")
st.sidebar.write(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')


st.title("ZeroWaste")
st.write("Coming soon: Image upload capability where you can upload an image to determine donation condition and best suited location through vision models" \
"for redistribution")
st.write("Enter a food item to determine if it's perishable or non-perishable: ")
food_input = st.text_input("Food Item: ")


if st.button("Check Food Type"):
    if food_input:
        test_vectorized = vectorizer.transform([food_input.lower()])
        pred = model.predict(test_vectorized)[0]

        if pred == 1:
            st.success(f'{food_input} is perishable')
            st.write(f"Check with local food pantry to determine if they accept {food_input}")
        else:
            st.info(f'{food_input} is non-perishable')
            st.write("Excellent for long time pantry donations")
st.markdown("---")


donation_csv = "donations_log.csv"
def load_donations() -> pd.DataFrame:
    if os.path.exists(donation_csv):
        df = pd.read_csv(donation_csv, parse_dates=["date"])
        return df
    return pd.DataFrame(columns=[
        "date", "item", "perishable", "quantity", "unit weight_lbs", "total_weight_lbs", "pantry", "zip"
    ])

def save_donations(df: pd.DataFrame):
    df.to_csv(donation_csv, index=False)

def estimate_weight(quantity: float, unit_weight_lbs: float)->float:
    try:
        return float(quantity) * float(unit_weight_lbs)
    except Exception:
        return 0.0

donations_df = load_donations()
tab_log, tab_find, tab_impact, tab_donate, tab_info = st.tabs(["📝 Log Donation", "🗺️ Find Food Bank", "📈 Impact", "💲Monetary Donation", "📊 Stats"])


with tab_log:
    st.header("Log A Donation")

    with st.form("donation_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            date = st.date_input("Date", value=dt.date.today())
            item = st.text_input("Food item (e.g., milk, canned beans)")
            qty = st.number_input("Quantity", min_value=1.0, step=1.0, value=1.0)
        with c2:
            unit_wt = st.number_input("Unit weight (lb)", min_value=0.0, step=0.1, value=1.0)
            pantry = st.text_input("Pantry / Food bank (optional)")
            zip_code = st.text_input("ZIP (optional)")

        #Logistic Regression prediction
        perishable_flag = None
        if item.strip():
            x = vectorizer.transform([item.lower()])
            perishable_flag = int(model.predict(x)[0])  # 1 = perishable, 0 = non-perishable

        if perishable_flag is not None:
            if perishable_flag == 1:
                st.success("Model says: **Perishable**")
            else:
                st.info("Model says: **Non-perishable**")

        submitted = st.form_submit_button("Save Donation")
        if submitted:
            if not item.strip():
                st.error("Please enter an item.")
            else:
                est_wt = estimate_weight(qty, unit_wt)
                new_row = {
                    "date": date.isoformat(),
                    "item": item.strip(),
                    "perishable": perishable_flag if perishable_flag is not None else None,
                    "quantity": qty,
                    "unit_weight_lb": unit_wt,
                    "est_weight_lb": est_wt,
                    "pantry": pantry.strip(),
                    "zip": zip_code.strip()
                }
                donations_df = pd.concat([donations_df, pd.DataFrame([new_row])], ignore_index=True)
                save_donations(donations_df)
                st.success("Donation saved!")

    # Show donations made in past
    if not donations_df.empty:
        st.subheader("Recent Donations")
        show_cols = ["date","item","perishable","quantity","unit_weight_lb","est_weight_lb","pantry","zip"]
        for c in show_cols:
            if c not in donations_df.columns:
                donations_df[c] = None
        st.dataframe(
            donations_df.sort_values("date", ascending=False)[show_cols],
            use_container_width=True
        )
        st.download_button(
            "Download donations (CSV)",
            data=donations_df[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="donations_log.csv",
            mime="text/csv"
        )

with tab_find:
    st.header("Locate Food Bank Closest To You")
    st.link_button(url ='https://www.feedingamerica.org/find-your-local-foodbank', label='Food Bank Locator')

with tab_impact:
    st.header("Donation Impact")
    if donations_df.empty:
        st.info("Log Donations in the 'Log Donation' Tab to see your donation impact!")
        #1.2 lbs donation is 1 lb meal (Feeding America) (1/1.2 = 0.83)
        #1 kg donation is approximately 2.65 kg of CO2 Emissions reduced (AFRA)
    else:
        colA, colB = st.columns(2)
        with colA:
            meals_per_lb = st.number_input("Meals Per Pound (0.83)", min_value=0.0, value=0.83, step=0.1)
        with colB:
            kgco2_per_lb = st.number_input("kg of CO2 avoided per pound (2.65)", min_value=0.0, value=2.65, step=0.1)
        view = donations_df.copy()
        if "est_weight_lb" not in view.columns and "total_weight_lbs" in view.columns:
            view["est_weight_lb"] = view["total_weight_lbs"]
        if "unit_weight_lb" not in view.columns and "unit weight_lbs" in view.columns:
            view["unit weight_lb"] = view["unit weight_lbs"]

        view["est_weight_lb"] = pd.to_numeric(view.get("est_weight_lb", 0), errors="coerce").fillna(0.0)
        view["quantity"] = pd.to_numeric(view.get("quantity", 0), errors="coerce").fillna(0.0)

        view["meals_est"] = (view["est_weight_lb"] * meals_per_lb).round(2)
        view["kg_co2_avoided_est"] = (view["est_weight_lb"] * kgco2_per_lb).round(2)

        total_items = int(view["quantity"].sum())
        total_weight = int(view["est_weight_lb"].sum())
        total_meals = int(view["meals_est"].sum())
        total_co2= int(view["kg_co2_avoided_est"].sum())

        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items Donated", f"{total_items}")
        m2.metric("Estimated Weight", f"{total_weight:,.1f} lb")
        m3.metric("Estimated Meals", f"{total_meals:,.0f}")
        m4.metric("CO₂ Emissions Avoided", f"{total_co2:,.1f} kg")

        st.markdown("Impact per Donation")
        show_cols = [
            "date","item","perishable","quantity","unit_weight_lb","est_weight_lb",
            "meals_est","kg_co2e_avoided_est","pantry","zip"
        ]
        for c in show_cols:
            if c not in view.columns:
                view[c] = None

        st.dataframe(
            view[show_cols].sort_values("date", ascending=False),
            use_container_width=True,
        )

        st.download_button(
            "Download impact table (CSV)",
            data=view[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="donations_impact.csv",
            mime="text/csv"
        )
with tab_donate:
    st.header("Monetary Donation")
    st.write("Each dollar donated becomes 10+ meals made and distributed")
    st.link_button(url ='https://give.feedingamerica.org/b34NMcohLUeT81zWoYT3Og2?oa_onsite_promo=homepage', label='Feeding America')
    st.link_button(url ='https://donate.foodbankiowa.org/give/738687/#!/donation/checkout?c_src=26FY910HPOG', label='Food Bank of Iowa')

with tab_info:
    st.header("Stats About Hunger")
    st.write("- 385,130 individuals in Iowa face hunger")
    st.write("- 120,000 children in Iowa face hunger" )
    st.write("- 1 in 8 people face hunger in Iowa")
    st.write("- 1 in 6 children face hunger in Iowa")
    st.write("- One-third of all food in the United States goes uneaten, and that in 2019, about 96 percent of households' wasted food ended up in landfills, combustion facilities, or down the drain to the sewer system (EPA) ")
    st.caption("Statistics by Feeding America and the U.S. Environmental Protection Agency")