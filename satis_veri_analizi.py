import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- 1. Her Ürün İçin Ayrı ve Sürekli Veri Üretimi ---
num_days = 365
np.random.seed(42)
dates = pd.date_range("2025-01-01", periods=num_days)

# Her bir ürün için kendi içinde tutarlı bir zaman serisi oluşturuyoruz.
sales_A = [10 + i//10 + np.random.randint(-3, 3) for i in range(num_days)]
sales_B = [25 - i//12 + np.random.randint(-3, 3) for i in range(num_days)]
sales_C = [15 + int(10*np.sin(i/20)) + np.random.randint(-2, 2) for i in range(num_days)]

# Bu ayrı serileri tek bir DataFrame'de birleştiriyoruz.
df_A = pd.DataFrame({'Date': dates, 'Product': 'A', 'Unit_sold': sales_A, 'Price': 20})
df_B = pd.DataFrame({'Date': dates, 'Product': 'B', 'Unit_sold': sales_B, 'Price': 30})
df_C = pd.DataFrame({'Date': dates, 'Product': 'C', 'Unit_sold': sales_C, 'Price': 50})

df = pd.concat([df_A, df_B, df_C]).sort_values(by='Date').reset_index(drop=True)

df["TotalSales"] = df["Unit_sold"] * df["Price"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)


# --- Analizler ---
# Aylık satışlar artık 12 ayın tamamını içerecek
monthly_sales = df["TotalSales"].resample("M").sum()
monthly_sales.index = monthly_sales.index.strftime('%B')
total_units = df.groupby("Product")["Unit_sold"].sum().sort_values(ascending=False)
product_sales = df.groupby("Product")["TotalSales"].sum().sort_values(ascending=False)

# --- Grafikler ---
# 1. Aylık Toplam Satış Grafiği (Tüm Yıl)
monthly_sales.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Aylık Toplam Satışlar (1 Yıl)', fontsize=16)
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



# --- Tahmin Modeli Kısmı ---
weekly_sales = df['TotalSales'].resample('W-MON').sum().reset_index()

# Özellik seti
weekly_sales['Rolling_Mean_4'] = weekly_sales['TotalSales'].shift(1).rolling(window=4).mean()
weekly_sales["Lag1"] = weekly_sales["TotalSales"].shift(1)
weekly_sales["Lag2"] = weekly_sales["TotalSales"].shift(2)
weekly_sales.dropna(inplace=True)

features = ["Rolling_Mean_4", "Lag1", "Lag2"]
target = "TotalSales"

X = weekly_sales[features]
y = weekly_sales[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Model
model = XGBRegressor(objective="reg:squarederror", 
                     n_estimators=100,
                     learning_rate=0.1,
                     max_depth=3)
                     
model.fit(X_train, y_train)

# --- Değerlendirme ---
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


print("\n" + "="*50)
print("Model Performansı")
print("="*50)
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")
print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.2f}")
print(f"R-Kare (R²) Skoru: {r2:.2f}")
if r2 > 0.5:
    print(f"-> Satışlardaki değişkenliğin yaklaşık %{r2*100:.0f}'ini açıklayabiliyor.")


# --- Tahmin Grafiği ---
plt.figure(figsize=(15, 7))
plt.plot(y_test.index, y_test, label="Gerçek Satışlar", marker=".", color='royalblue')
plt.plot(y_test.index, y_pred, label="XGBoost Tahmini", linestyle="--", marker=".", color="orange")
plt.legend()
plt.title("Haftalık Satış Tahmini (Test Verisi)", fontsize=16)
plt.show()
