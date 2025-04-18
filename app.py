import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Load trained model and dataset
model = joblib.load("sales_model.pkl")
df = pd.read_csv("train.csv", parse_dates=["date"])

# Streamlit App Title
st.title("🛍️ Sales Assistant for Stores")

# Tabs: Graph View and AI Chat Assistant
tab1, tab2 = st.tabs(["📈 Store Sales Graph", "🤖 AI Sales Prediction"])

# -------- Tab 1: Historical Sales Graph --------
with tab1:
    st.header("View Historical Sales")

    store_selection = st.selectbox("Select Store", df['store'].unique(), key="store_selection")
    item_selection = st.selectbox("Select Item", df['item'].unique(), key="item_selection")

    filtered_data = df[(df['store'] == store_selection) & (df['item'] == item_selection)]

    st.subheader(f"Sales Over Time for Store {store_selection}, Item {item_selection}")
    plt.figure(figsize=(12,6))
    filtered_data.groupby("date")["sales"].sum().plot()
    plt.title("Historical Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    st.pyplot(plt)

# -------- Tab 2: Future Sales Prediction --------
with tab2:
    st.header("Predict Future Sales")

    store_input = st.selectbox("Choose Store", df['store'].unique(), key="store_input")
    item_input = st.selectbox("Choose Item", df['item'].unique(), key="item_input")
    date_input = st.date_input("Choose a Date for Prediction", min_value=datetime.date.today(), key="date_input")

    def create_input_df(store, item, date):
        return pd.DataFrame([{
            'store': store,
            'item': item,
            'year': date.year,
            'month': date.month,
            'day': date.day,
            'dayofweek': date.weekday()
        }])

    # AI Chat-style Single Prediction
    if st.button("🤖 Predict Sales for Selected Date"):
        input_data = create_input_df(store_input, item_input, date_input)
        prediction = model.predict(input_data)[0]
        st.success(f"📅 On {date_input}, predicted sales = **{int(prediction)} units**")

    # Graph for Future Sales (Next 90 Days)
    st.subheader(f"📊 Future Sales Forecast (Next 90 Days) for Store {store_input}, Item {item_input}")
    future_dates = pd.date_range(datetime.date.today(), periods=90)
    predicted_sales = []

    for date in future_dates:
        input_data = create_input_df(store_input, item_input, date)
        predicted_sales.append(model.predict(input_data)[0])

    plt.figure(figsize=(12,6))
    plt.plot(future_dates, predicted_sales, label='Predicted Sales', color='orange')
    plt.title("Predicted Sales (Next 90 Days)")
    plt.xlabel("Date")
    plt.ylabel("Predicted Sales")
    plt.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(plt)
