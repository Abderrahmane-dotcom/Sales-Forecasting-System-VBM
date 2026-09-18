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
    Fill missing daily dates within each SalesHierarchy_PK between the first and last available dates 
    with QTY_CS = 0 while preserving existing observations.
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
    Replace zero values in the 'QTY_CS' column with the mean of their two preceding and two following non-zero neighbors."""
    values = group["QTY_CS"].to_numpy(dtype=float)

    zero_idx = np.where(values == 0)[0]

    for idx in zero_idx:

        before = values[max(0, idx-2):idx]
        after = values[idx+1:min(len(values), idx+3)]

        neighbors = np.concatenate([before, after])

        # Il faut avoir exactement 4 voisins
        if len(neighbors) != 4:
            continue

        # Si l'entourage contient un zéro, on passe
        if np.any(neighbors == 0):
            continue

        values[idx] = neighbors.mean()

    group["QTY_CS"] = values
    return group

def rolling_mean_std_from_start(x):
    """
    Calculate the rolling mean and standard deviation of 'QTY_CS' from the start of the series"""
    x["mean_shift1"] = x["QTY_CS"].expanding().mean().shift(1)
    x["std_shift1"] = x["QTY_CS"].expanding().std().shift(1)
    x["mean"] = x["QTY_CS"].expanding().mean()
    x["std"] = x["QTY_CS"].expanding().std()
    return x

def calculate_slope(row):
    """
    Calculate the slope of 'QTY_CS' values from lag 11 to lag 8 using linear regression.
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
    Calculate the slope of a linear regression line fitted to the given y-values.
    """
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



######## main function used in app.py (the above are helper functions) ########


def preprocess_for_inference(df, product_category, input_month):
    """
    Preprocess the input DataFrame for inference by creating lag features, handling missing values,
    and adding additional features such as Ramadan and Eid indicators."""

    #lets change imput_month to datetime and then to period
    input_month = pd.Period(input_month, freq="M")
    connection_string = (
        "mssql+pyodbc://@localhost/SalesDB"
        "?driver=ODBC+Driver+18+for+SQL+Server"
        "&trusted_connection=yes"
        "&TrustServerCertificate=yes"
    )

    
    engine = create_engine(connection_string)
    query = f"""
    SELECT DISTINCT SalesHierarchy_PK, S_UNIT
    FROM dbo.Sales
    WHERE Product_Category_Code = '{product_category}' and PK_date < '{input_month.start_time.date()}'
    """

    df_original_unique = pd.read_sql(query, engine)

    engine.dispose()

    df["year_month"] = df["PK_Date"].dt.to_period("M")
    monthly_df = (
        df
        .groupby(["SalesHierarchy_PK", "year_month"], as_index=False)["QTY_CS"]
        .sum()
    )


    df = monthly_df.copy()
    del monthly_df
    gc.collect()
    df = df.merge(df_original_unique, on="SalesHierarchy_PK", how="left")

    # the first try to avoid the 0 and nulls for inference
    df = (
        df.groupby("SalesHierarchy_PK", group_keys=False)
        .apply(replace_zeros_with_neighbors)
    )

    df.reset_index(drop=True, inplace=True)
    df["year"] = df["year_month"].dt.year
    df["month"] = df["year_month"].dt.month

    lags = [1, 2, 3, 8, 9, 10, 11, 12]
    
    rows = []

    # Loop over each SalesHierarchy_PK
    for pk in df["SalesHierarchy_PK"].unique():
        row_data = {
            "SalesHierarchy_PK": pk,
            "year_month": input_month
        }
        # Data for this PK only
        pk_df = df[df["SalesHierarchy_PK"] == pk]

        for lag in lags:

            target_month = input_month - lag
            true_value = 0
            value = np.nan
            # Try current year then previous years, we could optimize this later
            for years_back in range(0, 5):

                candidate_month = target_month - 12*years_back

                match = pk_df.loc[
                    (pk_df["year_month"] == candidate_month) &
                    (pk_df["QTY_CS"] != 0),
                    "QTY_CS"
                ]

                if years_back == 0 :
                    true_value = match.iloc[0] if not match.empty else 0
                if not match.empty:
                    value = match.iloc[0]
                    break
                    
            if pd.isna(value):
                for years_back in range(0, 5):
                    candidate_month = target_month - 12*years_back
                    # Get S_UNIT of current PK
                    s_unit = pk_df["S_UNIT"].iloc[0]
                    # Only if S_UNIT exists
                    if pd.notna(s_unit):
                                                
                        sunit_match = df.loc[
                            (df["S_UNIT"] == s_unit) &
                            (df["year_month"] == candidate_month)&
                            (df["QTY_CS"] != 0),
                            "QTY_CS"
                        ]

                        if not sunit_match.empty:
                            value = sunit_match.mean()
                            break
            if pd.isna(value):
                for years_back in range(0, 5):
                    candidate_month = target_month - 12*years_back


                    sunit_match = df.loc[
                        (df["year_month"] == candidate_month)&
                        (df["QTY_CS"] != 0),
                        "QTY_CS"
                    ]

                    if not sunit_match.empty:
                        value = sunit_match.mean()
                        break      
                            
            # Final assignment
            row_data[f"QTY_CS_lag_{lag}"] = value
            row_data[f"QTY_CS_lag_true_value_{lag}"] = true_value
        row_data["year"] = input_month.year
        row_data["month"] = input_month.month
        row_data["S_UNIT"] = pk_df["S_UNIT"].iloc[0]
        


        rows.append(row_data)

    # Final dataframe
    df_inference = pd.DataFrame(rows)
    # Add Ramadan and Eid features

    df_inference["ramadan_ratio"] = df_inference["year_month"].apply(ramadan_ratio)
    df_inference["is_eid_al_fitr"] = df_inference["year_month"].apply(is_eid_al_fitr)
    df_inference["is_eid_al_adha"] = df_inference["year_month"].apply(is_eid_al_adha)

    for lag in lags:
        df_inference[f"QTY_CS_lag_original_{lag}"] = df_inference[f"QTY_CS_lag_{lag}"]

    #lets now normalise the data with the mean and std

    stats = (
        df.groupby("SalesHierarchy_PK")["QTY_CS"]
        .agg(mean="mean", std="std")
    )

    df_inference["mean"] = df_inference["SalesHierarchy_PK"].map(stats["mean"])
    df_inference["std"] = df_inference["SalesHierarchy_PK"].map(stats["std"])

    
    def non_zero_count(pk):
        return (df[df["SalesHierarchy_PK"] == pk]["QTY_CS"] != 0).sum()
    def mean_std_from_lags(row):
        pk = row["SalesHierarchy_PK"]
        if non_zero_count(pk) < 13:
            # calculate mean and std from lags
            lag_values = [row[f"QTY_CS_lag_original_{lag}"] for lag in lags]
            lag_values = [v for v in lag_values if not pd.isna(v)]
            if len(lag_values) > 0:
                row["mean"] = np.mean(lag_values)
                row["std"] = np.std(lag_values)
        return row
    df_inference = df_inference.apply(mean_std_from_lags, axis=1)



    df_inference["is_summer"] = int(input_month.month in [6, 7, 8])
    df_inference["month_sin"] = np.sin(2 * np.pi * input_month.month / 12)
    df_inference["month_cos"] = np.cos(2 * np.pi * input_month.month / 12)

    for lag in lags : 
        df_inference[f"QTY_CS_lag_{lag}"] = (df_inference[f"QTY_CS_lag_{lag}"] - df_inference["mean"])/df_inference["std"]

    
    
    # maintenant on doit créer les autres features 
    df_inference["slope_lag_11_to_8"] = df_inference.apply(calculate_slope, axis=1)
    df_inference["mean_lag_11_to_8"] = df_inference[["QTY_CS_lag_11", "QTY_CS_lag_10", "QTY_CS_lag_9", "QTY_CS_lag_8"]].mean(axis=1)
    

    df_missing_routes = df_original_unique[~df_original_unique["SalesHierarchy_PK"].isin(df_inference["SalesHierarchy_PK"])]
    # now lets sum the count of unique SalesHierarchy_PK in df_missing_routes
    missing_routes_count = df_missing_routes["SalesHierarchy_PK"].nunique()
    print("-------------------------------")
    print("the number of routes not token to account in prediction (not enough data) :",missing_routes_count)
    print("-------------------------------")
    #and now the count of unique SalesHierarchy_PK in the inference data
    inference_routes_count = df_inference["SalesHierarchy_PK"].nunique()
    print("the number of routes taken into consideration in the model : ",inference_routes_count)


   
    return df_inference
    


    


