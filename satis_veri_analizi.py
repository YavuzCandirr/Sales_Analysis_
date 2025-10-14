import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#Veri oluşturma kısmı
Products = np.random.choice(["A", "B", "C"], size=180)
data = {
    "Date": pd.date_range("2025-01-01", periods=180).tolist(),
    "Product": Products,
    "Unit_sold": [
        10 + i//10 + np.random.randint(-3,3) if p == "A" else
        25 - i//12 + np.random.randint(-3,3) if p == "B" else
        15 + int(3*np.sin(i/5)) + np.random.randint(-2,2)
        for i, p in enumerate(Products)
    ],
    "Price": [
        20 if p=="A" else 30 if p=="B" else 50 for p in Products
    ]
}

df = pd.DataFrame(data)
df["TotalSales"] = df["Unit_sold"] * df["Price"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# --- Analizler ---
monthly_sales = df["TotalSales"].resample("M").sum()
monthly_sales.index = monthly_sales.index.strftime('%B')
total_units = df.groupby("Product")["Unit_sold"].sum().sort_values(ascending=False)
product_sales = df.groupby("Product")["TotalSales"].sum().sort_values(ascending=False)

# --- Grafikler ---
# 1. Aylık Toplam Satış Grafiği (İlk 3 Ay)
monthly_sales.head(3).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('İlk 3 Ayın Toplam Satışları', fontsize=16)
plt.xlabel('Aylar', fontsize=12)
plt.ylabel('Toplam Gelir', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Ürün Bazında Toplam Gelir Grafiği
product_sales.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Ürün Bazında Toplam Gelir', fontsize=16)
plt.xlabel('Ürün', fontsize=12)
plt.ylabel('Toplam Gelir', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Ürün Bazında Satılan Birim Adedi Grafiği
total_units.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Ürün Bazında Satılan Toplam Birim', fontsize=16)
plt.xlabel('Ürün', fontsize=12)
plt.ylabel('Satılan Birim Sayısı', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- XGBoost ile Tahmin Modeli Kısmı ---
weekly_sales = df['TotalSales'].resample('W-MON').sum().reset_index()
weekly_sales["Week"] = weekly_sales["Date"].dt.isocalendar().week
weekly_sales["Month"] = weekly_sales["Date"].dt.month
weekly_sales["Lag1"] = weekly_sales["TotalSales"].shift(1)
weekly_sales["Lag2"] = weekly_sales["TotalSales"].shift(2)
weekly_sales["Lag3"] = weekly_sales["TotalSales"].shift(3)
weekly_sales.dropna(inplace=True)

features = ["Week", "Month", "Lag1", "Lag2", "Lag3"]
target = "TotalSales"
X = weekly_sales[features]
y = weekly_sales[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
model = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X_train, y_train)

# Tahmin döngüsü...
forecasts = []
n_forecast = 4
last_row = weekly_sales.iloc[-1]
last_date = last_row['Date']
last_lags = [last_row['TotalSales'], last_row['Lag1'], last_row['Lag2']]
for i in range(n_forecast):
    future_date = last_date + pd.Timedelta(days=(i + 1) * 7)
    future_week = future_date.isocalendar().week
    future_month = future_date.month
    current_features = pd.DataFrame([{"Week": future_week, "Month": future_month, "Lag1": last_lags[0], "Lag2": last_lags[1], "Lag3": last_lags[2]}], columns=features)
    pred = model.predict(current_features)[0]
    forecasts.append(pred)
    last_lags = [pred] + last_lags[:-1]

# Tahmin Grafiği
plt.figure(figsize=(12, 6))
plt.plot(weekly_sales["Date"], weekly_sales["TotalSales"], label="Gerçek Satışlar", marker="o", color='royalblue')
future_dates = pd.date_range(weekly_sales["Date"].iloc[-1] + pd.Timedelta(days=7), periods=n_forecast, freq="7D")
plt.plot(future_dates, forecasts, label="XGBoost Tahmini", linestyle="--", marker="x", color="orange", markersize=8)
plt.legend()
plt.title("XGBoost Haftalık Satış Tahmini", fontsize=16)
plt.xlabel("Tarih", fontsize=12)
plt.ylabel("Toplam Satış", fontsize=12)
plt.grid(True)
plt.show()



# Modelin test verisi üzerindeki tahminlerini yapalım
y_pred = model.predict(X_test)

# Metrikleri hesaplayalım
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("XGBoost Model Performans Değerlendirmesi")
print("="*50)
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")
print(f"-> Tahminlerimiz gerçek satışlardan ortalama {mae:.2f} birim sapıyor.")
print("-" * 50)
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.2f}")
print("-> Bu, büyük hatalara daha duyarlı bir sapma ölçüsüdür.")
print("-" * 50)
print(f"R-Kare (R²) Skoru: {r2:.2f}")
print(f"-> Modelimiz, satışlardaki değişkenliğin yaklaşık %{r2*100:.0f}'ini açıklayabiliyor.")
print("="*50 + "\n")