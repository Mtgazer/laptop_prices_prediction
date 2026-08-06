import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻"
)

st.title("💻 Laptop Price Predictor")

# Load model and features
model = joblib.load("model_1.pkl")
features = joblib.load("features.pkl")

# Load dataset (used only to populate dropdowns)
df = pd.read_csv("laptopPrice.csv")  
gb_columns = [
    "ram_gb",
    "ssd",
    "hdd",
    "graphic_card_gb"
]

for col in gb_columns:
    df[col] = (
        df[col]
        .str.replace(" GB", "", regex=False)
        .astype(int)
    )

df["warranty"] = (
    df["warranty"]
    .replace({
        "No warranty": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3
    })
    .astype(int)
)   

# =========================
# User Inputs
# =========================

brand = st.selectbox("Brand", sorted(df["brand"].unique()))

processor_brand = st.selectbox(
    "Processor Brand",
    sorted(df["processor_brand"].unique())
)

processor_name = st.selectbox(
    "Processor Name",
    sorted(df["processor_name"].unique())
)

processor_gnrtn = st.selectbox(
    "Processor Generation",
    sorted(df["processor_gnrtn"].unique())
)

ram_gb = st.selectbox(
    "RAM (GB)",
    sorted(df["ram_gb"].unique())
)

ram_type = st.selectbox(
    "RAM Type",
    sorted(df["ram_type"].unique())
)

ssd = st.selectbox(
    "SSD (GB)",
    sorted(df["ssd"].unique())
)

hdd = st.selectbox(
    "HDD (GB)",
    sorted(df["hdd"].unique())
)

graphic_card_gb = st.selectbox(
    "Graphics Card (GB)",
    sorted(df["graphic_card_gb"].unique())
)

os = st.selectbox(
    "Operating System",
    sorted(df["os"].unique())
)

os_bit = st.selectbox(
    "OS Bit",
    sorted(df["os_bit"].unique())
)

weight = st.selectbox(
    "Weight Category",
    sorted(df["weight"].unique())
)

warranty = st.selectbox(
    "Warranty",
    sorted(df["warranty"].unique())
)

rating = st.slider("Rating", 1, 5, 4)

touchscreen = st.checkbox("Touchscreen")

msoffice = st.checkbox("MS Office")

number_ratings = st.number_input(
    "Number of Ratings",
    min_value=0,
    value=100
)

number_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    value=50
)

# =========================
# Prediction
# =========================

if st.button("Predict Price"):

    # Create empty dataframe
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=features
    )

    # Numerical features
    input_df["ram_gb"] = ram_gb
    input_df["ssd"] = ssd
    input_df["hdd"] = hdd
    input_df["graphic_card_gb"] = graphic_card_gb
    input_df["warranty"] = warranty
    input_df["rating"] = rating
    input_df["Touchscreen"] = int(touchscreen)
    input_df["msoffice"] = int(msoffice)
    input_df["Number of Ratings"] = number_ratings
    input_df["Number of Reviews"] = number_reviews

    # Encode categorical features
    categorical_values = {
        "brand": brand,
        "processor_brand": processor_brand,
        "processor_name": processor_name,
        "processor_gnrtn": processor_gnrtn,
        "ram_type": ram_type,
        "os": os,
        "os_bit": os_bit,
        "weight": weight
    }

    for prefix, value in categorical_values.items():
        column = f"{prefix}_{value}"

        if column in input_df.columns:
            input_df[column] = 1

    # Predict
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(f"💰 Estimated Laptop Price: EGP {prediction:,.2f}")
