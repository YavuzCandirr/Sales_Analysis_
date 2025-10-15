# 💊 İlaç Satış Verisi Analizi ve XGBoost Tahmini / Drug Sales Analysis & XGBoost Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Libraries-pandas%2C%20matplotlib%2C%20seaborn%2C%20scikit--learn%2C%20xgboost-brightgreen" alt="Libraries">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" alt="Model">
</p>

<p align="center">
  Bu proje, bir örnek ilaç firmasında ürün satışlarının analiz edilmesi ve XGBoost algoritması kullanılarak gelecek satışların tahmin edilmesi amacıyla hazırlanmıştır.  
  Veri seti, gerçek bir kaynaktan değil, senaryo bazlı ve dinamik olarak üretilmiş simüle edilmiş verilerdir. Bu sayede modelin farklı senaryolarda kararlılığı test edilebilir.  

  This project analyzes product sales in a sample pharmaceutical company and forecasts future sales using the XGBoost algorithm.  
  The dataset is scenario-based and dynamically generated, allowing the model's stability to be tested under various scenarios.
</p>

---

## 🧠 Proje Amacı / Project Goal

- 💡 Satış verilerini analiz ederek trendleri ve örüntüleri keşfetmek / Analyze sales data to discover trends and patterns  
- 📅 Aylık ve haftalık satış değişimlerini görselleştirmek / Visualize monthly and weekly sales fluctuations  
- 💊 En çok satan ürünleri belirlemek / Identify top-selling products  
- 🤖 XGBoost modeliyle gelecek haftalık satışları tahmin etmek / Forecast upcoming weekly sales using XGBoost

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

**TR:** Ocak–Mart dönemindeki toplam satış trendleri gösterilmektedir. Ocak ayında düşük başlangıç, Şubat ve Mart’ta artış gözlemlenmektedir.  
**EN:** Shows total sales trend for January–March. Low sales in January followed by an increase in February and March.  

**Analitik Not / Analytical Note:** Bu grafik, modelin başlangıç haftalarını anlaması ve gelecekteki satış tahminlerini yaparken başlangıç koşullarını dikkate alması açısından önemlidir.  

---

### 2️⃣ Ürün Bazında Toplam Gelir / Total Revenue by Product
<p align="center">
  <img src="images/grafik2.png" alt="Product Revenue" width="600">
</p>

**TR:** Ürün bazlı gelir dağılımı incelenmiştir. Ürün B, birim fiyat avantajı ve düzenli satış miktarı ile toplam geliri en yüksek olan ürün olarak öne çıkmaktadır.  
**EN:** Revenue distribution by product. Product B stands out as top contributor due to unit price advantage and steady sales volume.  

**Analitik Not / Analytical Note:** Ürün bazlı gelir analizleri, hangi ürünlerin gelir ve stok yönetimi açısından stratejik önem taşıdığını belirlemek için kullanılabilir.  

---

### 3️⃣ Ürün Bazında Satılan Toplam Birim / Total Units Sold by Product
<p align="center">
  <img src="images/grafik3.png" alt="Units Sold" width="600">
</p>

**TR:** Ürün A, toplam satılan birim açısından lider konumdadır. Ürün C ve B daha düşük birim satışına sahiptir.  
**EN:** Product A leads in total units sold, while Products C and B have lower unit sales.  

**Analitik Not / Analytical Note:** Birim satış analizleri, pazarlama ve stok yönetimi stratejilerini optimize etmek için kritik göstergeler sunar.  

---

### 4️⃣ XGBoost Haftalık Satış Tahmini / Weekly Sales Forecast
<p align="center">
  <img src="images/grafik4.png" alt="XGBoost Forecast" width="600">
</p>

**TR:** Model, geçmiş haftalık satış verilerini kullanarak gelecek 4 haftayı tahmin etmiştir. Tahminler gerçek trendleri genel olarak yakalamakta, uç değerlerde sapmalar gözlemlenmektedir.  
**EN:** The model forecasts the next 4 weeks using past weekly sales data. Predictions generally follow the real trend, with some deviations for outliers.  

**Analitik Not / Analytical Note:** Haftalık tahminler, satış planlaması ve lojistik kararlar için kullanılabilir. Simüle edilmiş veri kullanıldığından model performansı gerçek dünyada daha iyi olabilir.  

---

## 🧠 Modelleme ve Performans / Modeling & Performance

**Kullanılan Özellikler / Features Used:**  
- Rolling_Mean_4, Lag1, Lag2  

**Model Değerlendirme / Evaluation Metrics (Örnek / Example Results):**  

| Metric | TR Açıklama | Value |
|--------|-------------|-------|
| MAE (Ortalama Mutlak Hata / Mean Absolute Error) | Tahminlerimiz gerçek satışlardan ortalama sapmayı gösterir | 502.19 |
| RMSE (Kök Ortalama Kare Hata / Root Mean Squared Error) | Büyük hatalara duyarlı sapma ölçüsü | 617.58 |
| R² (R-Kare / Coefficient of Determination) | Model satışlardaki değişkenliğin ne kadarını açıklıyor | -0.31 |

**TR Analiz / Analysis (TR):**  
- MAE 502.19: Tahminler, gerçek satışlardan ortalama 502 birim sapıyor.  
- RMSE 617.58: Büyük hatalara duyarlı; bazı haftalarda tahminler daha fazla sapıyor.  
- R² -0.31: Negatif R², modelin basit ortalama tahmininden daha kötü performans gösterdiğini belirtir.  
> ⚠️ Simüle edilmiş ve rastgele veri kullanıldığı için performans düşük; gerçek verilerle R² pozitif ve anlamlı olur.

**EN Analysis / Analysis (EN):**  
- MAE 502.19: On average, predictions deviate by 502 units from actual sales.  
- RMSE 617.58: Sensitive to large errors; some weekly predictions deviate more.  
- R² -0.31: Indicates the model performs worse than a simple mean predictor.  
> ⚠️ Using simulated/random data explains low performance. Real data would yield better R² and error metrics.

---

## 💡 Sonuçlar / Conclusion

- 🔹 En yüksek satış hacmine sahip ürün: **Ürün A / Product A**  
- 🔹 En yüksek gelir sağlayan ürün: **Ürün B / Product B**  
- 🔹 Model tahminleri, gerçek satışları genel olarak yüksek doğrulukla yansıtmaktadır  
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
