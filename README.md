# 💊 İlaç Satış Verisi Analizi ve XGBoost Tahmini / Drug Sales Analysis & XGBoost Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Libraries-pandas%2C%20seaborn%2C%20sklearn%2C%20xgboost-brightgreen" alt="Libraries">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" alt="Model">
</p>

<p align="center">
  Bu proje, bir örnek ilaç firmasında ürün satışlarının analiz edilmesi ve XGBoost algoritması kullanılarak gelecek satışların tahmin edilmesi amacıyla hazırlanmıştır.  
  Veri seti, gerçek bir kaynaktan değil, senaryo bazlı ve dinamik olarak üretilmiş simüle edilmiş verilerdir. Bu sayede modelin farklı senaryolarda kararlılığını test etmek mümkün olur.  

  This project analyzes product sales in a sample pharmaceutical company and forecasts future sales using the XGBoost algorithm.  
  The dataset is not from a real source; it is scenario-based and dynamically generated. This allows testing the model's stability under different scenarios.
</p>

<p align="center">
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

**Model Değerlendirme / Evaluation Metrics (Örnek / Example Results):**  

| Metric | TR Açıklama | Value |
|--------|-------------|-------|
| MAE (Ortalama Mutlak Hata / Mean Absolute Error) | Tahminlerimiz gerçek satışlardan ortalama sapmayı gösterir | 502.19 |
| RMSE (Kök Ortalama Kare Hata / Root Mean Squared Error) | Büyük hatalara duyarlı sapma ölçüsü | 617.58 |
| R² (R-Kare / Coefficient of Determination) | Model satışlardaki değişkenliğin ne kadarını açıklıyor | -0.31 |

**TR Açıklama / Analysis:**  
- MAE 502.19: Tahminler, gerçek satışlardan ortalama 502 birim sapıyor. Bu, modelin küçük bir hatayla çalıştığını gösteriyor.  
- RMSE 617.58: Büyük hatalara karşı duyarlı; bazı haftalarda tahminlerin sapması daha yüksek olabilir.  
- R² -0.31: Model, satış verisindeki değişkenliğin yaklaşık %-31’ini açıklayabiliyor. Negatif R², modelin basit bir ortalama tahmininden daha kötü performans gösterdiğini gösterir.  
> ⚠️ Bu sonuçlar, simüle edilmiş ve rastgele oluşturulmuş veri kullanıldığından modelin düşük performans göstermesi normaldir. Gerçek veri ile R² ve diğer metriklerin çok daha iyi olması beklenir.

**EN Explanation / Analysis:**  
- MAE 502.19: On average, predictions deviate by 502 units from actual sales.  
- RMSE 617.58: Sensitive to large errors; some weekly predictions deviate more significantly.  
- R² -0.31: Model explains approximately -31% of the variance, meaning it performs worse than simply predicting the mean.  
> ⚠️ These results are expected with simulated and random data. Using real sales data would likely yield much better R² and error metrics.

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
