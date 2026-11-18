import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="House Price Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 15px;
}

div[data-testid="stMetric"] {
    border: 1px solid #d9d9d9;
    padding: 15px;
    border-radius: 12px;
    background-color: #fafafa;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🏠 House Price Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'House price analysis, market insights and machine learning prediction'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD ORIGINAL DATA
# =========================================================

@st.cache_data
def load_original_data():

    df = pd.read_csv("house_pred_modified_data.csv")

    df["date"] = pd.to_datetime(df["date"])

    return df


# =========================================================
# PREPROCESS DATA
# =========================================================

@st.cache_data
def prepare_data():

    df = pd.read_csv("house_pred_modified_data.csv")

    # Remove zero-price records
    df = df[df["price"] > 0].copy()

    # Date processing
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    df.drop("date", axis=1, inplace=True)

    # Remove unnecessary/leakage columns
    df.drop(
        ["street", "price_per_sqft"],
        axis=1,
        inplace=True
    )

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=["city", "statezip"],
        drop_first=True
    )

    return df


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = prepare_data()

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        mae,
        mse,
        rmse,
        r2
    )


# =========================================================
# LOAD DATA + MODEL
# =========================================================

original_df = load_original_data()

df = prepare_data()

(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    mae,
    mse,
    rmse,
    r2
) = train_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎛️ Dashboard Controls")

st.sidebar.markdown("### Market Filters")

cities = sorted(original_df["city"].dropna().unique())

selected_city = st.sidebar.selectbox(
    "Select City",
    ["All Cities"] + cities
)

bedroom_options = sorted(
    original_df["bedrooms"].dropna().unique()
)

selected_bedrooms = st.sidebar.selectbox(
    "Bedrooms",
    ["All"] + [str(int(x)) for x in bedroom_options]
)

st.sidebar.divider()

st.sidebar.info(
    "Use the filters to explore the housing market. "
    "Prediction inputs are available in the Prediction tab."
)


# =========================================================
# FILTER DATA
# =========================================================

filtered_df = original_df.copy()

if selected_city != "All Cities":

    filtered_df = filtered_df[
        filtered_df["city"] == selected_city
    ]

if selected_bedrooms != "All":

    filtered_df = filtered_df[
        filtered_df["bedrooms"]
        == int(selected_bedrooms)
    ]


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Price Prediction",
        "🤖 Model Performance",
        "📋 Data Explorer"
    ]
)


# =========================================================
# TAB 1 - DASHBOARD
# =========================================================

with tab1:

    st.markdown(
        '<div class="section-title">📊 Housing Market Overview</div>',
        unsafe_allow_html=True
    )

    if len(filtered_df) > 0:

        average_price = filtered_df["price"].mean()

        median_price = filtered_df["price"].median()

        average_living = filtered_df[
            "sqft_living"
        ].mean()

        total_houses = len(filtered_df)

        # -------------------------------------------------
        # KPI CARDS
        # -------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "🏠 Houses",
                f"{total_houses:,}"
            )

        with c2:

            st.metric(
                "💰 Average Price",
                f"${average_price:,.0f}"
            )

        with c3:

            st.metric(
                "📌 Median Price",
                f"${median_price:,.0f}"
            )

        with c4:

            st.metric(
                "📐 Avg Living Area",
                f"{average_living:,.0f} sqft"
            )

        st.divider()

        # -------------------------------------------------
        # PRICE DISTRIBUTION
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("💰 Price Distribution")

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.hist(
                filtered_df["price"],
                bins=30,
                edgecolor="black"
            )

            ax.set_xlabel("House Price")

            ax.set_ylabel("Number of Houses")

            ax.set_title("Distribution of House Prices")

            st.pyplot(fig)

        # -------------------------------------------------
        # BEDROOM ANALYSIS
        # -------------------------------------------------

        with col2:

            st.subheader("🛏️ Price by Bedrooms")

            bedroom_price = (
                filtered_df
                .groupby("bedrooms")["price"]
                .mean()
                .sort_index()
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(
                bedroom_price.index.astype(str),
                bedroom_price.values
            )

            ax.set_xlabel("Bedrooms")

            ax.set_ylabel("Average Price")

            ax.set_title("Average House Price by Bedrooms")

            st.pyplot(fig)

        # -------------------------------------------------
        # LIVING AREA VS PRICE
        # -------------------------------------------------

        st.subheader("📐 Living Area vs House Price")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            filtered_df["sqft_living"],
            filtered_df["price"],
            alpha=0.4
        )

        ax.set_xlabel("Living Area (sqft)")

        ax.set_ylabel("House Price")

        ax.set_title(
            "Relationship Between Living Area and Price"
        )

        st.pyplot(fig)

        # -------------------------------------------------
        # CITY ANALYSIS
        # -------------------------------------------------

        st.subheader("🌆 Average Price by City")

        city_price = (
            original_df
            .groupby("city")["price"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(
            city_price.index,
            city_price.values
        )

        ax.set_xlabel("City")

        ax.set_ylabel("Average Price")

        ax.set_title("Top Cities by Average House Price")

        ax.tick_params(axis="x", rotation=45)

        st.pyplot(fig)

        # -------------------------------------------------
        # HOUSE AGE
        # -------------------------------------------------

        st.subheader("🏗️ House Age vs Price")

        age_df = filtered_df.copy()

        age_df["house_age"] = (
            2026 - age_df["yr_built"]
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            age_df["house_age"],
            age_df["price"],
            alpha=0.35
        )

        ax.set_xlabel("House Age (Years)")

        ax.set_ylabel("House Price")

        ax.set_title(
            "Relationship Between House Age and Price"
        )

        st.pyplot(fig)

    else:

        st.warning("No houses match the selected filters.")


# =========================================================
# TAB 2 - PRICE PREDICTION
# =========================================================

with tab2:

    st.subheader("🔮 House Price Prediction")

    st.write(
        "Enter the property details below to estimate the house price."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )

        sqft_living = st.number_input(
            "Living Area (sqft)",
            min_value=100,
            max_value=10000,
            value=1800
        )

        sqft_lot = st.number_input(
            "Lot Area (sqft)",
            min_value=100,
            max_value=50000,
            value=5000
        )

    with col2:

        floors = st.number_input(
            "Floors",
            min_value=1.0,
            max_value=4.0,
            value=1.0,
            step=0.5
        )

        waterfront = st.selectbox(
            "Waterfront",
            [0, 1]
        )

        view = st.selectbox(
            "View Quality",
            sorted(
                original_df["view"].dropna().unique()
            )
        )

        condition = st.selectbox(
            "Condition",
            sorted(
                original_df["condition"].dropna().unique()
            )
        )

    with col3:

        sqft_above = st.number_input(
            "Above Ground Area (sqft)",
            min_value=100,
            max_value=10000,
            value=1500
        )

        sqft_basement = st.number_input(
            "Basement Area (sqft)",
            min_value=0,
            max_value=5000,
            value=300
        )

        yr_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=2026,
            value=2000
        )

        yr_renovated = st.number_input(
            "Year Renovated",
            min_value=0,
            max_value=2026,
            value=0
        )

    col4, col5 = st.columns(2)

    with col4:

        prediction_city = st.selectbox(
            "City",
            sorted(
                original_df["city"].dropna().unique()
            )
        )

    with col5:

        prediction_statezip = st.selectbox(
            "State ZIP",
            sorted(
                original_df["statezip"].dropna().unique()
            )
        )

    st.divider()

    if st.button(
        "🔮 Predict House Price",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "bedrooms": [bedrooms],

            "bathrooms": [bathrooms],

            "sqft_living": [sqft_living],

            "sqft_lot": [sqft_lot],

            "floors": [floors],

            "waterfront": [waterfront],

            "view": [view],

            "condition": [condition],

            "sqft_above": [sqft_above],

            "sqft_basement": [sqft_basement],

            "yr_built": [yr_built],

            "yr_renovated": [yr_renovated],

            "year": [2026],

            "month": [11]
        })

        # Add encoded city columns

        for col in X_train.columns:

            if col.startswith("city_"):

                input_data[col] = (
                    1
                    if col == f"city_{prediction_city}"
                    else 0
                )

        # Add encoded statezip columns

        for col in X_train.columns:

            if col.startswith("statezip_"):

                input_data[col] = (
                    1
                    if col == f"statezip_{prediction_statezip}"
                    else 0
                )

        # Match training columns exactly

        input_data = input_data.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            "Prediction generated successfully!"
        )

        st.markdown("### Estimated House Price")

        st.metric(
            "Predicted Price",
            f"${prediction:,.2f}"
        )

        # Prediction summary

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Bedrooms",
                bedrooms
            )

        with c2:

            st.metric(
                "Living Area",
                f"{sqft_living:,} sqft"
            )

        with c3:

            st.metric(
                "Bathrooms",
                bathrooms
            )


# =========================================================
# TAB 3 - MODEL PERFORMANCE
# =========================================================

with tab3:

    st.subheader("🤖 Machine Learning Model Performance")

    st.info(
        "The selected final model is Linear Regression. "
        "It achieved the highest R² score among the tested models."
    )

    # KPI metrics

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    with c2:

        st.metric(
            "MAE",
            f"${mae:,.0f}"
        )

    with c3:

        st.metric(
            "RMSE",
            f"${rmse:,.0f}"
        )

    with c4:

        st.metric(
            "Test Data",
            f"{len(y_test):,}"
        )

    st.divider()

    # ------------------------------------------------------
    # MODEL COMPARISON
    # ------------------------------------------------------

    st.subheader("🏆 Model Comparison")

    comparison = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Gradient Boosting",
            "Random Forest",
            "Decision Tree"
        ],

        "R² Score": [
            0.696696,
            0.693920,
            0.690645,
            0.673681,
            -1.223241,
            -4.702614
        ],

        "RMSE": [
            212420.98,
            213390.75,
            214529.26,
            220332.85,
            575110.86,
            921075.51
        ]
    })

    comparison = comparison.sort_values(
        "R² Score",
        ascending=False
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    # R2 graph

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        comparison["Model"],
        comparison["R² Score"]
    )

    ax.set_xlabel("Model")

    ax.set_ylabel("R² Score")

    ax.set_title(
        "R² Score Comparison"
    )

    ax.tick_params(
        axis="x",
        rotation=35
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    st.pyplot(fig)

    # ------------------------------------------------------
    # ACTUAL VS PREDICTED
    # ------------------------------------------------------

    st.subheader("🎯 Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    ax.set_xlabel(
        "Actual Price"
    )

    ax.set_ylabel(
        "Predicted Price"
    )

    ax.set_title(
        "Actual vs Predicted House Prices"
    )

    st.pyplot(fig)

    # ------------------------------------------------------
    # RESIDUAL ANALYSIS
    # ------------------------------------------------------

    st.subheader("📉 Residual Analysis")

    residuals = y_test - y_pred

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        y_pred,
        residuals,
        alpha=0.5
    )

    ax.axhline(
        y=0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Predicted Price"
    )

    ax.set_ylabel(
        "Residual"
    )

    ax.set_title(
        "Residual Plot"
    )

    st.pyplot(fig)


# =========================================================
# TAB 4 - DATA EXPLORER
# =========================================================

with tab4:

    st.subheader("📋 Dataset Explorer")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Original Records",
            f"{len(original_df):,}"
        )

    with c2:

        st.metric(
            "Usable Records",
            f"{len(df):,}"
        )

    with c3:

        st.metric(
            "Features",
            f"{df.shape[1] - 1}"
        )

    st.divider()

    # Dataset preview

    st.subheader("🔎 Dataset Preview")

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )

    # Statistical summary

    st.subheader("📊 Statistical Summary")

    st.dataframe(
        filtered_df.describe(),
        use_container_width=True
    )

    # Download

    csv_data = filtered_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_house_data.csv",
        mime="text/csv"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    **🏠 House Price Analytics Dashboard**

    Machine Learning Project using Linear Regression

    Dataset: House Price Dataset  
    Final Model: Linear Regression  
    R² Score: 0.6967
    """
)