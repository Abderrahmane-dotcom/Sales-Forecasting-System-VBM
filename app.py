from pendulum import datetime
from datetime import datetime
import streamlit as st
from trainer import *
from inference import *


st.title("Sales Forecasting")

product_category = st.selectbox(
    "Product Category",
    [
        "CSD",
        "WATER",
        "ENERGY",
        "LIPTON",
        "SNACKS"
    ]
)

year_month = st.text_input(
    "Year-Month",
    placeholder="AAAA-MM"
)

if st.button("Process"):

    with st.spinner("Generating forecast..."):

        end_date = datetime.strptime(f"{year_month}-01", "%Y-%m-%d").date()

        df = get_data(product_category, end_date)

        df_train = process_for_training(df)

        lag_list = [1,2,3,8,9,10,11,12]
        feature_cols = [f"QTY_CS_lag_{lag}" for lag in lag_list] + [
            "is_summer","month_sin","month_cos","slope_lag_11_to_8","mean_lag_11_to_8","ramadan_ratio","is_eid_al_fitr","is_eid_al_adha"]
        model = train_xgboost(
            df_train[feature_cols],
            df_train["QTY_CS_scaled"],
            "models/xgb_best_params.json",
            product_category
        )
        df_inference = preprocess_for_inference(df, product_category, year_month)

        
        preds = model.predict(df_inference[feature_cols])
        df_inference["predicted_QTY_CS_scaled"] = preds
        df_inference["predicted_QTY_CS"] = df_inference["predicted_QTY_CS_scaled"] * df_inference["std"] + df_inference["mean"]

        df_inference["predicted_QTY_CS"] = (
            df_inference["predicted_QTY_CS"]
            .round()
            .astype(int)
        )
        prediction_month = pd.Period(year_month, freq="M")
        month_lags = []
        for lag in lag_list:
            month_lag = prediction_month - lag
            month_lags.append(month_lag)
            df_inference.rename(
                columns={
                    f"QTY_CS_lag_true_value_{lag}": f"QTY_CS_{month_lag}"
                },
                inplace=True,
            )

        result_final = df_inference[["SalesHierarchy_PK", "year_month", "predicted_QTY_CS","S_UNIT"]+[f"QTY_CS_{month}" for month in month_lags]]

    st.success("Done!")
    st.dataframe(result_final)
    from io import BytesIO

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result_final.to_excel(writer, index=False)

    st.download_button(
        label="Download Excel",
        data=buffer.getvalue(),
        file_name=f"{product_category}_{year_month}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )