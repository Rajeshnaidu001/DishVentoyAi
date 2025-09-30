
import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# –––––––––– Config ––––––––––
excel_path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSKW_3bL3LB7U9kDYZN207PFEnDLULYiQ3Zczv-clcLXNkJY_0kj-71WE_EiEB16bzUkGd6mVCvLoxK/pub?output=xlsx"  # TODO: set your file
sales_sheet = "pizza_sales"
weights_sheet = "ingredient_weights"
horizon_days = 7
# Holidays (extend as needed)
holidays = pd.DataFrame({
    'holiday': ['christmas', 'new_year'],
    'ds': pd.to_datetime(['2015-12-25', '2016-01-01']),
    'lower_window': 0,
    'upper_window': 1,
})

# Optional evaluation split
train_end_date = pd.Timestamp('2015-12-21')
test_end_date = pd.Timestamp('2015-12-28')

# -------------------- Load data --------------------
df = pd.read_excel(excel_path, sheet_name=sales_sheet)

# Validate and clean sales
if 'order_date' not in df.columns:
    raise ValueError("pizza_sales sheet must include 'order_date' column.")
need_cols_sales = {'pizza_id', 'quantity', 'pizza_size', 'pizza_category'}
missing_sales = need_cols_sales - set(df.columns)
if missing_sales:
    raise ValueError(f"pizza_sales sheet missing columns: {missing_sales}")

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df = df.dropna(subset=['order_date', 'pizza_id', 'quantity', 'pizza_size', 'pizza_category'])

# Load ingredient weights
weights = pd.read_excel(excel_path, sheet_name=weights_sheet)

# Detect weights sheet format
has_per_pizza = {'pizza_name', 'pizza_category', 'pizza_size', 'ingredient', 'quantity_per_pizza'}.issubset(weights.columns)
has_category_size = {'ingredient', 'pizza_category', 'pizza_size', 'quantity_per_pizza'}.issubset(weights.columns)

if not (has_per_pizza or has_category_size):
    raise ValueError("ingredient_weights must have either columns "
                     "[pizza_name, pizza_category, pizza_size, ingredient, quantity_per_pizza] "
                     "or [ingredient, pizza_category, pizza_size, quantity_per_pizza].")

weights = weights.copy()
weights['quantity_per_pizza'] = pd.to_numeric(weights['quantity_per_pizza'], errors='coerce')
if has_per_pizza:
    weights = weights.dropna(subset=['pizza_name','pizza_category','pizza_size','ingredient','quantity_per_pizza'])
else:
    weights = weights.dropna(subset=['ingredient','pizza_category','pizza_size','quantity_per_pizza'])

# Build lookup for pizza metadata
lookup_cols = ['pizza_id', 'pizza_size', 'pizza_category']
if 'pizza_name' in df.columns:
    lookup_cols.append('pizza_name')
menu_lookup = df[lookup_cols].drop_duplicates().dropna(subset=['pizza_id','pizza_size','pizza_category'])

# -------------------- Optional evaluation on 7-day test window --------------------
results = []
pizza_forecasts_eval = {}

pizza_ids = df['pizza_id'].unique()

for pizza in tqdm(pizza_ids, desc='Evaluating pizzas'):
    pizza_df = df[df['pizza_id'] == pizza]
    daily_sales = pizza_df.groupby('order_date')['quantity'].sum().reset_index()
    daily_sales.rename(columns={'order_date': 'ds', 'quantity': 'y'}, inplace=True)

    if len(daily_sales) < 10:
        continue

    train = daily_sales[daily_sales['ds'] <= train_end_date]
    test = daily_sales[(daily_sales['ds'] > train_end_date) & (daily_sales['ds'] <= test_end_date)]
    if len(test) == 0:
        continue

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=15.0,
        holidays=holidays
    )
    try:
        model.fit(train)
    except Exception as e:
        print(f"Skipping {pizza} due to fit error: {e}")
        continue

    periods = (test_end_date - train_end_date).days
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)

    forecast_test = forecast[forecast['ds'].isin(test['ds'])]
    merged = test.set_index('ds').join(
        forecast_test.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]
    ).dropna()
    if merged.empty:
        continue

    mae = mean_absolute_error(merged['y'], merged['yhat'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
    denom = merged['y'].replace(0, np.nan)
    mape = (np.abs(merged['y'] - merged['yhat']) / denom).mean() * 100

    results.append({
        'pizza_id': pizza,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'test_days': len(merged),
        'train_days': len(train)
    })
    pizza_forecasts_eval[pizza] = merged

if results:
    results_df = pd.DataFrame(results)
    print("\nEvaluation preview (first 20 rows):")
    print(results_df.head(20))
    print(f"\nOverall average MAE: {results_df['mae'].mean():.2f}")
    print(f"Overall average RMSE: {results_df['rmse'].mean():.2f}")
    print(f"Overall average MAPE: {results_df['mape'].dropna().mean():.2f}%")
else:
    print("\nNo evaluation metrics computed.")

# -------------------- Production 7-day forecast --------------------
seven_day_pizza_forecasts = []  # rows with ds, yhat, yhat_lower, yhat_upper, pizza_id

for pizza in tqdm(pizza_ids, desc='7-day forecasting'):
    pizza_df = df[df['pizza_id'] == pizza]
    daily_sales = pizza_df.groupby('order_date')['quantity'].sum().reset_index()
    daily_sales.rename(columns={'order_date': 'ds', 'quantity': 'y'}, inplace=True)

    if len(daily_sales) < 10:
        continue

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=15.0,
        holidays=holidays
    )
    try:
        model.fit(daily_sales)
    except Exception as e:
        print(f"Skipping {pizza} due to fit error: {e}")
        continue

    future = model.make_future_dataframe(periods=horizon_days, freq='D')
    future = future[future['ds'] > daily_sales['ds'].max()]
    if future.empty:
        continue

    forecast = model.predict(future)
    f = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    f['pizza_id'] = pizza
    seven_day_pizza_forecasts.append(f)

if not seven_day_pizza_forecasts:
    raise RuntimeError("No 7-day forecasts produced. Check data sufficiency and model fitting.")

seven_day_pizza_forecasts = pd.concat(seven_day_pizza_forecasts, ignore_index=True)

# -------------------- Export pizza forecasts as JSON (daily and totals) --------------------
daily_basic = seven_day_pizza_forecasts.sort_values(["ds", "pizza_id"])[
    ["ds", "pizza_id", "yhat", "yhat_lower", "yhat_upper"]
].copy()
daily_basic["ds"] = pd.to_datetime(daily_basic["ds"]).dt.strftime("%Y-%m-%d")
daily_basic.to_json("pizza_forecasts_7day_daily.json", orient="records", indent=2)

weekly_bounds = (
    seven_day_pizza_forecasts.groupby("pizza_id", as_index=False)
    .agg(
        yhat_7day_total=("yhat", "sum"),
        yhat_7day_lower=("yhat_lower", "sum"),
        yhat_7day_upper=("yhat_upper", "sum"),
    )
    .sort_values("yhat_7day_total", ascending=False)
)
weekly_bounds.to_json("pizza_forecasts_7day_totals.json", orient="records", indent=2)

# With metadata if pizza_name exists
if 'pizza_name' in df.columns:
    menu_lookup_full = df[["pizza_id", "pizza_name", "pizza_size", "pizza_category"]].drop_duplicates()
    daily_with_meta = seven_day_pizza_forecasts.merge(menu_lookup_full, on="pizza_id", how="left")
    daily_with_meta = daily_with_meta[
        ["ds", "pizza_id", "pizza_name", "pizza_size", "pizza_category", "yhat", "yhat_lower", "yhat_upper"]
    ].sort_values(["ds", "pizza_name", "pizza_size"]).copy()
    daily_with_meta["ds"] = pd.to_datetime(daily_with_meta["ds"]).dt.strftime("%Y-%m-%d")
    daily_with_meta.to_json("pizza_forecasts_7day_daily_with_meta.json", orient="records", indent=2)

    weekly_with_meta = (
        daily_with_meta.groupby(["pizza_id", "pizza_name", "pizza_size", "pizza_category"], as_index=False)
        .agg(
            yhat_7day_total=("yhat", "sum"),
            yhat_7day_lower=("yhat_lower", "sum"),
            yhat_7day_upper=("yhat_upper", "sum"),
        )
        .sort_values(["pizza_name", "pizza_size"])
    )
    weekly_with_meta.to_json("pizza_forecasts_7day_totals_with_meta.json", orient="records", indent=2)

# -------------------- Map to ingredient needs --------------------
# Attach metadata to forecasts
fcast_with_meta = seven_day_pizza_forecasts.merge(menu_lookup, on='pizza_id', how='left')
if fcast_with_meta[['pizza_size', 'pizza_category']].isna().any().any():
    missing_ids = fcast_with_meta[
        fcast_with_meta['pizza_size'].isna() | fcast_with_meta['pizza_category'].isna()
    ]['pizza_id'].unique()
    raise ValueError(f"Missing pizza_size/category for pizza_ids: {missing_ids[:10]} ...")

# Choose join keys depending on weights format
if has_per_pizza:
    if 'pizza_name' not in fcast_with_meta.columns:
        raise ValueError("Weights are per pizza_name but pizza_sales has no pizza_name column.")
    join_keys = ['pizza_name', 'pizza_category', 'pizza_size']
else:
    join_keys = ['pizza_category', 'pizza_size']

segmented = fcast_with_meta.merge(weights, on=join_keys, how='inner')
if segmented.empty:
    raise ValueError("Join with ingredient_weights produced no rows. "
                     "Ensure ingredient_weights covers all observed (pizza_name/category/size) combinations.")

# Compute ingredient demand from forecast bands
for col_in, col_out in [('yhat', 'qty'), ('yhat_lower', 'qty_lower'), ('yhat_upper', 'qty_upper')]:
    segmented[col_out] = segmented[col_in].clip(lower=0) * segmented['quantity_per_pizza']

# Aggregate ingredient demand
ingredient_daily = (
    segmented.groupby(['ds', 'ingredient'], as_index=False)
             .agg(ingredient_qty=('qty', 'sum'),
                  ingredient_qty_lower=('qty_lower', 'sum'),
                  ingredient_qty_upper=('qty_upper', 'sum'))
             .sort_values(['ds', 'ingredient'])
)
ingredient_daily_out = ingredient_daily.copy()
ingredient_daily_out["ds"] = pd.to_datetime(ingredient_daily_out["ds"]).dt.strftime("%Y-%m-%d")
ingredient_daily_out.to_json('ingredient_daily_7day.json', orient='records', indent=2)

ingredient_7day_total = (
    ingredient_daily.groupby('ingredient', as_index=False)
                    .agg(total=('ingredient_qty', 'sum'),
                         total_lower=('ingredient_qty_lower', 'sum'),
                         total_upper=('ingredient_qty_upper', 'sum'))
                    .sort_values('ingredient')
)
ingredient_7day_total.to_json('ingredient_7day_totals.json', orient='records', indent=2)

ingredient_by_size = (
    segmented.groupby(['ds', 'pizza_size', 'ingredient'], as_index=False)
             .agg(qty=('qty', 'sum'),
                  qty_lower=('qty_lower', 'sum'),
                  qty_upper=('qty_upper', 'sum'))
)
ingredient_by_size_out = ingredient_by_size.copy()
ingredient_by_size_out["ds"] = pd.to_datetime(ingredient_by_size_out["ds"]).dt.strftime("%Y-%m-%d")
ingredient_by_size_out.to_json('ingredient_by_size_7day.json', orient='records', indent=2)

ingredient_by_category = (
    segmented.groupby(['ds', 'pizza_category', 'ingredient'], as_index=False)
             .agg(qty=('qty', 'sum'),
                  qty_lower=('qty_lower', 'sum'),
                  qty_upper=('qty_upper', 'sum'))
)
ingredient_by_category_out = ingredient_by_category.copy()
ingredient_by_category_out["ds"] = pd.to_datetime(ingredient_by_category_out["ds"]).dt.strftime("%Y-%m-%d")
ingredient_by_category_out.to_json('ingredient_by_category_7day.json', orient='records', indent=2)

print("\nSaved JSON files:")
print("- pizza_forecasts_7day_daily.json")
print("- pizza_forecasts_7day_totals.json")
if 'pizza_name' in df.columns:
    print("- pizza_forecasts_7day_daily_with_meta.json")
    print("- pizza_forecasts_7day_totals_with_meta.json")
print("- ingredient_daily_7day.json")
print("- ingredient_7day_totals.json")
print("- ingredient_by_size_7day.json")
print("- ingredient_by_category_7day.json")
