import json
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import gc
import pandas as pd
from hijri_converter import convert
import calendar
from datetime import date, timedelta
import xgboost as xgb



### helper functions
def fill_daily(group):
    """
    Fill missing daily dates within each SalesHierarchy_PK between the first and last available dates.
    Missing days are added with QTY_CS = 0 while preserving existing observations.
    """
    group = group.sort_values('PK_Date')
    
    
    full_range = pd.date_range(
        start=group['PK_Date'].min(),
        end=group['PK_Date'].max(),
        freq='D'
    )
    
    group = group.set_index('PK_Date').reindex(full_range)

    
    group['SalesHierarchy_PK'] = group['SalesHierarchy_PK'].iloc[0]
    group['QTY_CS'] = group['QTY_CS'].fillna(0)
    
    group.index.name = 'PK_Date'
    
    return group.reset_index()



def replace_zeros_with_neighbors(group):
    """
    Replace zero values in the 'QTY_CS' column with the mean of their two preceding and two following non-zero neighbors.
    """
    values = group["QTY_CS"].to_numpy(dtype=float)

    zero_idx = np.where(values == 0)[0]

    for idx in zero_idx:

        before = values[max(0, idx-2):idx]
        after = values[idx+1:min(len(values), idx+3)]

        neighbors = np.concatenate([before, after])

        
        if len(neighbors) != 4:
            continue

        
        if np.any(neighbors == 0):
            continue

        values[idx] = neighbors.mean()

    group["QTY_CS"] = values
    return group



def rolling_mean_std_from_start(x):
    """
    Calculate the rolling mean and standard deviation of the 'QTY_CS' column from the start of the series.
    """
    x["mean_shift1"] = x["QTY_CS"].expanding().mean().shift(1)
    x["std_shift1"] = x["QTY_CS"].expanding().std().shift(1)
    x["mean"] = x["QTY_CS"].expanding().mean()
    x["std"] = x["QTY_CS"].expanding().std()
    return x



def calculate_slope(row):
    """
    calculate the slope of the linear regression line fitted to the last four lagged values of 'QTY_CS'.
    """
    y = [row["QTY_CS_lag_11"], row["QTY_CS_lag_10"], row["QTY_CS_lag_9"], row["QTY_CS_lag_8"]]
    x = [1, 2, 3, 4]
    if any(pd.isnull(y)):
        return np.nan
    else:
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
def slope(y):
    """
    Calculate the slope of the linear regression line fitted to the values in y."""
    if len(y) < 2:
        return 0

    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]





def ramadan_ratio(year_month):
    """
    Returns the proportion of days in the Gregorian month
    that belong to Ramadan.
    """
    year = year_month.year
    month = year_month.month

    total_days = calendar.monthrange(year, month)[1]
    ramadan_days = 0

    for day in range(1, total_days + 1):
        hijri = convert.Gregorian(year, month, day).to_hijri()

        if hijri.month == 9:
            ramadan_days += 1

    return ramadan_days / total_days



def is_eid_al_fitr(year_month):
    """
    Returns 1 if the Gregorian month contains Eid al-Fitr
    (1 Shawwal), otherwise 0.
    """
    year = year_month.year
    month = year_month.month

    total_days = calendar.monthrange(year, month)[1]

    for day in range(1, total_days + 1):
        hijri = convert.Gregorian(year, month, day).to_hijri()

        if hijri.month == 10 and hijri.day == 1:
            return 1

    return 0


def is_eid_al_adha(year_month):
    """
    Returns 1 if the Gregorian month contains Eid al-Adha
    (10 Dhu al-Hijjah), otherwise 0.
    """
    year = year_month.year
    month = year_month.month

    total_days = calendar.monthrange(year, month)[1]

    for day in range(1, total_days + 1):
        hijri = convert.Gregorian(year, month, day).to_hijri()

        if hijri.month == 12 and hijri.day == 10:
            return 1

    return 0



######## main functions used in app.py (the above are helper functions) ########

def get_data(product_category,end_date):
    """Connect to the SQL Server database and retrieve data for the specified product category and end date."""

    connection_string = (
        "mssql+pyodbc://@localhost/SalesDB"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&trusted_connection=yes"
        "&TrustServerCertificate=yes"
    )

    
    engine = create_engine(connection_string)

    query = text("""
    SELECT SalesHierarchy_PK, PK_date, SUM(QTY_CS) as QTY_CS FROM dbo.Sales
    WHERE Product_Category_Code = :product_category AND PK_date < :end_date
    GROUP BY SalesHierarchy_PK, PK_date ORDER BY SalesHierarchy_PK, PK_date
    """)

    df = pd.read_sql(
        query,
        engine,
        params={"product_category": product_category, "end_date": end_date}
    )
    engine.dispose()
    return df






def process_for_training(df):
    """Process the raw data for training by filling missing dates, aggregating monthly, and creating features."""

    df['PK_Date'] = pd.to_datetime(df['PK_date'])
    

    df = df.groupby("SalesHierarchy_PK", group_keys=False).apply(fill_daily)
    
    #df is now filled daily


    df["year_month"] = df["PK_Date"].dt.to_period("M")
    monthly_df = (
        df
        .groupby(["SalesHierarchy_PK", "year_month"], as_index=False)["QTY_CS"]
        .sum()
    )

    df = monthly_df.copy()
    del monthly_df
    gc.collect()

    df = (
        df.groupby("SalesHierarchy_PK", group_keys=False)
        .apply(replace_zeros_with_neighbors)
    )

    keep_pks = []

    for pk, subset in df.groupby("SalesHierarchy_PK"):
        total_count = len(subset)
        zero_count = (subset["QTY_CS"] == 0).sum()

        if zero_count == 0 and total_count >= 13:
            keep_pks.append(pk)

    df = df[df["SalesHierarchy_PK"].isin(keep_pks)]
    df = df.groupby("SalesHierarchy_PK",group_keys=False).apply(rolling_mean_std_from_start)
    df["QTY_CS_scaled"] = (df["QTY_CS"] - df["mean"]) / df["std"]

    lag_list = [1,2,3,8,9,10,11,12]
    for lag in lag_list:
        df[f"QTY_CS_lag_{lag}"] = (
            df.groupby("SalesHierarchy_PK")["QTY_CS"]
            .shift(lag)
        )
    for lag in lag_list:
        df[f"QTY_CS_lag_{lag}"] = (df[f"QTY_CS_lag_{lag}"]-df["mean_shift1"])/df["std_shift1"]

    df["slope_lag_11_to_8"] = df.apply(calculate_slope, axis=1)
    df["mean_lag_11_to_8"] = df[["QTY_CS_lag_11", "QTY_CS_lag_10", "QTY_CS_lag_9", "QTY_CS_lag_8"]].mean(axis=1)
    

    df = df.dropna()

    df.reset_index(drop=True, inplace=True)

    # here we'll use the hijri converter to build the hijri date features
    df["ramadan_ratio"] = df["year_month"].apply(ramadan_ratio)

    df["is_eid_al_fitr"] = (
        df["year_month"]
        .apply(is_eid_al_fitr)
        .astype("int")
    )

    df["is_eid_al_adha"] = (
        df["year_month"]
        .apply(is_eid_al_adha)
        .astype("int")
    )
    df["is_summer"] = df["year_month"].dt.month.isin([6, 7, 8]).astype("int")
    df["month_sin"] = np.sin(2 * np.pi * df["year_month"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["year_month"].dt.month / 12)

    return df



def train_xgboost(X_train, y_train, params_path, product_category):
    """Train an XGBoost model using the provided training data and hyperparameters."""
    # Load tuned hyperparameters
    with open(params_path, "r") as f:
        params = json.load(f)

    # change quantile_alpha closer to 1 for higher tendency to predict higher values, and closer to 0 for lower values
    params.update({
        "objective": "reg:quantileerror",
        "quantile_alpha": 0.85,
        "tree_method": "hist",   
        "random_state": 42,
        "n_jobs": -1,
    })

    # Create model
    model = xgb.XGBRegressor(**params)

    # Train
    model.fit(X_train, y_train)
    model.save_model(f"models/xgb_model_{product_category}.json")
    return model


