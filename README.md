# 📈 İlaç Satış Analizi ve XGBoost Tahmini / Drug Sales Analysis & XGBoost Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Libraries-Pandas%2C%20XGBoost%2C%20Sklearn-brightgreen" alt="Libraries">
  <img src="https://img.shields.io/badge/Model-XGBoost-orange" alt="Model">
</p>

<p align="center">
  Bu proje, bir örnek ilaç firmasında ürün satışlarının analiz edilmesi ve XGBoost algoritması kullanılarak gelecek satışların tahmin edilmesi amacıyla hazırlanmıştır.
  Veri seti, her ürünün kendi içinde tutarlı bir zaman serisi deseni izlediği, senaryo bazlı ve dinamik olarak üretilmiş simüle edilmiş verilerdir. Bu yaklaşım, modelin tahmin yeteneğini test etmek için ideal bir ortam sağlar.

  This project analyzes product sales in a sample pharmaceutical company and forecasts future sales using the XGBoost algorithm.
  The dataset is scenario-based and dynamically generated, with each product following its own consistent time-series pattern, providing an ideal environment for testing the model's forecasting capabilities.
</p>

---

## 🧠 Proje Amacı / Project Goal

- 💡 Tutarlı desenlere sahip sentetik satış verisi üreterek analiz ve modelleme yapmak / Generate and analyze synthetic sales data with consistent patterns for modeling
- 📅 Aylık ve ürün bazlı satış dinamiklerini görselleştirmek / Visualize monthly and product-based sales dynamics
- 🛠️ Etkili özellik mühendisliği (Feature Engineering) ile model performansını artırmak / Enhance model performance through effective feature engineering
- 🤖 XGBoost modeliyle gelecek haftalık satışları yüksek doğrulukla tahmin etmek / Forecast upcoming weekly sales with high accuracy using XGBoost

---

## ⚙️ Kullanılan Teknolojiler / Technologies

| Alan / Area | Teknoloji / Technology |
|-------------|-----------------------|
| Programlama Dili / Language | Python |
| Kütüphaneler / Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost |
| Model / Model | XGBoost Regressor |
| Ortam / Environment | Jupyter Notebook / Google Colab / VS Code |

---

## 🧩 Veri Seti Bilgileri / Dataset Info

| Sütun / Column | Açıklama / Description |
|----------------|----------------------|
| Date | Satış tarihi / Sale date |
| Product | Ürün adı / Product name (A, B, C) |
| Unit_sold | Satılan miktar / Units sold |
| Price | Birim fiyat / Unit price |
| TotalSales | Toplam satış geliri / Total sales revenue |

---

## 📊 Veri Analizi ve Görselleştirmeler / Visualizations

### 1️⃣ Yıllık Toplam Satışlar / Total Annual Sales
<p align="center">
  <img src="images/grafik1.png" alt="Monthly Sales" width="600">
</p>

**TR:** 12 aylık dönemdeki toplam satış gelirleri gösterilmektedir. Grafikte, farklı ürünlerin kendi trendlerinin (artan, azalan ve mevsimsel) birleşik etkisi gözlemlenmektedir.
**EN:** Shows the total sales revenue over a 12-month period. The graph reflects the combined effect of the individual trends (increasing, decreasing, and seasonal) of the different products.

**Analitik Not / Analytical Note:** Bu genel bakış, yıllık bütçeleme ve genel talep planlaması için temel bir göstergedir. Yıl içindeki zirve ve dip noktaları, stratejik kararlar için önemli ipuçları verir.

---

### 2️⃣ Ürün Bazında Toplam Gelir / Total Revenue by Product
<p align="center">
  <img src="images/grafik2.png" alt="Product Revenue" width="600">
</p>

**TR:** Ürün C, yüksek birim fiyatı sayesinde en yüksek toplam geliri sağlayan ürün olarak öne çıkmaktadır. Ürün B ve A onu takip etmektedir.
**EN:** Product C stands out as the top revenue contributor, thanks to its high unit price. It is followed by Products B and A.

**Analitik Not / Analytical Note:** Gelir analizi, hangi ürünlerin kârlılık açısından en kritik olduğunu belirler. Yüksek gelirli ürünlerin tedarik zinciri ve pazarlama stratejilerine öncelik verilebilir.

---

### 3️⃣ Ürün Bazında Satılan Toplam Birim / Total Units Sold by Product
<p align="center">
  <img src="images/grafik3.png" alt="Units Sold" width="600">
</p>

**TR:** Ürün B, toplamda en çok satılan birim adedine sahiptir. Bu, ürünün pazardaki popülaritesini ve talebinin yüksek olduğunu gösterir.
**EN:** Product B leads in the total number of units sold, indicating its popularity and high demand in the market.

**Analitik Not / Analytical Note:** Birim satış ve gelir analizleri birlikte değerlendirildiğinde, "sürümden kazandıran" (yüksek hacim, düşük kâr marjı) ve "yüksek kârlı" (düşük hacim, yüksek kâr marjı) ürünler ayırt edilebilir.

---

## 🧠 Modelleme ve Performans / Modeling & Performance

**Kullanılan Özellikler / Features Used:**
- `Rolling_Mean_4`: Son 4 haftanın satış ortalaması, modele yakın geçmişteki trendi hatırlatır.
- `Lag1`, `Lag2`: Bir ve iki hafta önceki satış değerleri, anlık değişimleri yakalamak için kullanılır.

**Model Değerlendirme / Evaluation Metrics:**

| Metric | TR Açıklama | Value |
|--------|-------------|-------|
| MAE (Ortalama Mutlak Hata) | Tahminlerin ortalama sapma miktarı | 450.75 |
| RMSE (Kök Ortalama Kare Hata) | Büyük hatalara daha duyarlı sapma ölçüsü | 580.21 |
| R² (R-Kare) | Modelin verideki değişkenliği açıklama yüzdesi | 0.92 |

**TR Analiz / Analysis (TR):**
- MAE & RMSE: Hata metriklerinin düşük olması, modelin tahminlerinin gerçek değerlere oldukça yakın olduğunu gösterir.
- R² 0.92: Çok yüksek bir R-Kare skoru! Bu, modelimizin satışlardaki değişkenliğin **%92'sini** başarıyla açıklayabildiğini gösterir.
> ✅ Bu yüksek performans, her ürün için tutarlı desenler içeren doğru veri üretim mantığı ve etkili özellik mühendisliği (`Rolling_Mean`) sayesinde elde edilmiştir.

**EN Analysis / Analysis (EN):**
- MAE & RMSE: The low error metrics indicate that the model's predictions are very close to the actual values.
- R² 0.92: A very high R-squared score! This shows that our model can successfully explain **92%** of the variance in sales.
> ✅ This high performance was achieved thanks to the correct data generation logic with consistent patterns for each product and effective feature engineering (`Rolling_Mean`).

---

### 🤖 XGBoost Haftalık Satış Tahmini / Weekly Sales Forecast
<p align="center">
  <img src="images/grafik4.png" alt="XGBoost Forecast" width="600">
</p>

**TR:** Model, daha önce görmediği test verileri üzerinde yüksek isabetli tahminler yapmıştır. Mavi (gerçek) ve turuncu (tahmin) çizgilerin neredeyse üst üste gitmesi, modelin başarısını görsel olarak kanıtlamaktadır.
**EN:** The model made highly accurate predictions on unseen test data. The close overlap between the blue (actual) and orange (predicted) lines visually confirms the model's success.

---

## 💡 Sonuçlar / Conclusion

- 🔹 En yüksek satış hacmine sahip ürün: **Ürün B / Product B**
- 🔹 En yüksek gelir sağlayan ürün: **Ürün C / Product C**
- 🔹 Doğru veri simülasyonu ve etkili özellik mühendisliği, tahmin modelinin performansını kökten iyileştirmiştir.
- 🔹 XGBoost modeli, haftalık satışları yüksek doğrulukla tahmin etme potansiyeline sahiptir.

---

### 👨‍💻 Geliştirici / Developer
<p align="center">
  <b>Yavuz ÇANDIR</b>
  <br>
  <a href="https://github.com/YavuzCandirr" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-YavuzCandirr-black?logo=github" alt="GitHub">
  </a>
</p>
