# 💊 İlaç Satış Verisi Analizi ve XGBoost Tahmini / Drug Sales Analysis & XGBoost Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Libraries-pandas%2C%20seaborn%2C%20sklearn%2C%20xgboost-brightgreen" alt="Libraries">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" alt="Model">
</p>

<p align="center">
  Bu proje, <b>ilaç satış verilerini analiz etmek</b> ve <b>gelecekteki satışları XGBoost ile tahmin etmek</b> amacıyla hazırlanmıştır.  
  Python veri bilimi araçları ile <b>trend analizi</b>, <b>ürün bazlı satış incelemeleri</b> ve <b>makine öğrenmesi tahminleri</b> gerçekleştirilmiştir.
</p>

---

## 🧠 Proje Amacı / Project Goal

- 💡 Satış verilerini analiz ederek <b>trendleri ve örüntüleri keşfetmek</b> / Discover trends and patterns in sales data  
- 📅 Aylık ve haftalık satış değişimlerini görselleştirmek / Visualize monthly and weekly sales changes  
- 💊 En çok satan ürünleri belirlemek / Identify top-selling products  
- 🤖 XGBoost modeliyle <b>gelecek haftalık satışları tahmin etmek</b> / Forecast future weekly sales using XGBoost

---

## ⚙️ Kullanılan Teknolojiler / Technologies

| Alan / Area | Teknoloji / Technology |
|-------------|-----------------------|
| Programlama Dili / Language | Python |
| Kütüphaneler / Libraries | pandas, matplotlib, seaborn, scikit-learn, xgboost |
| Model / Model | XGBoost |
| Ortam / Environment | Jupyter Notebook / Google Colab |

---

## 🧩 Veri Seti Bilgileri / Dataset Info

| Sütun / Column | Açıklama / Description |
|----------------|----------------------|
| Date | Satış tarihi / Sale date |
| Product | Ürün adı / Product name |
| Unit_sold | Satılan miktar / Units sold |
| Price | Birim fiyat / Unit price |
| TotalSales | Toplam satış / Total sales |

---

## 📊 Veri Analizi ve Görselleştirmeler / Visualizations

### 1️⃣ İlk 3 Ayın Toplam Satışları / Total Sales of First 3 Months
<p align="center">
  <img src="images/grafik1.png" alt="Monthly Sales" width="600">
</p>

**TR:** Ocak–Mart döneminde toplam satışların dağılımı gösterilmektedir. Trend artışı gözlemlenmektedir.  
**EN:** Shows the distribution of total sales from January to March. An upward trend is observed.

---

### 2️⃣ Ürün Bazında Toplam Gelir / Total Revenue by Product
<p align="center">
  <img src="images/grafik2.png" alt="Product Revenue" width="600">
</p>

**TR:** Her ürünün toplam gelire katkısı incelenmiştir. Ürün B, birim fiyat avantajıyla yüksek gelir sağlamıştır.  
**EN:** Examines each product’s contribution to total revenue. Product B generates high revenue due to unit price advantage.

---

### 3️⃣ Ürün Bazında Satılan Toplam Birim / Total Units Sold by Product
<p align="center">
  <img src="images/grafik3.png" alt="Units Sold" width="600">
</p>

**TR:** Her ürünün toplam satış adedi karşılaştırılmıştır. Ürün A en yüksek satış hacmine sahiptir.  
**EN:** Compares total units sold per product. Product A has the highest sales volume.

---

### 4️⃣ XGBoost Haftalık Satış Tahmini / Weekly Sales Forecast
<p align="center">
  <img src="images/grafik4.png" alt="XGBoost Forecast" width="600">
</p>

**TR:** Model, geçmiş haftalık satış verilerini kullanarak önümüzdeki 4 haftayı tahmin etmiştir. Tahminler gerçek trendleri büyük oranda yakalamaktadır.  
**EN:** The model forecasts the next 4 weeks using past weekly sales data. Predictions closely follow the real trends.

---

## 🧠 Modelleme ve Performans / Modeling & Performance

**Kullanılan Özellikler / Features Used:**  
- Week, Month, Lag1, Lag2, Lag3  

**Model Değerlendirme / Evaluation Metrics:**  
- MAE (Ortalama Mutlak Hata / Mean Absolute Error)  
- RMSE (Kök Ortalama Kare Hata / Root Mean Squared Error)  
- R² (R-Kare / Coefficient of Determination)  

**TR:** Model satışlardaki değişkenliğin yaklaşık %84’ünü açıklamaktadır.  
**EN:** The model explains approximately 84% of the variance in sales.

---

## 💡 Sonuçlar / Conclusion

- 🔹 En yüksek satış hacmine sahip ürün: **Ürün A / Product A**  
- 🔹 En yüksek gelir sağlayan ürün: **Ürün B / Product B**  
- 🔹 Model tahminleri, gerçek satışları yüksek doğrulukla yansıtmaktadır  
- 🔹 Gelecek çalışmalar: Ek değişkenler ekleyerek tahmin doğruluğu artırılabilir  

---

### 👨‍💻 Geliştirici / Developer
<p align="center">
  <b>Yavuz ÇANDIR</b>  
  <br>
  <a href="https://github.com/YavuzCandirr" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-YavuzCandirr-black?logo=github" alt="GitHub">
  </a>
</p>
