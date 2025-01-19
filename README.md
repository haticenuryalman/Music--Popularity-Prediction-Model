# Music--Popularity-Prediction-Model

Proje: Müzik Veri Seti ile Popülerlik Tahmini

Bu proje, bir müzik veri seti kullanılarak parçaların popülerlik değerlerini tahmin etmeyi amaçlamaktadır.Bir müzik veri seti üzerinde popülerlik tahmini yapmak için çeşitli makine öğrenmesi modelleri kullanılarak gerçekleştirilmiştir. 

Kullanılan veri seti: https://www.kaggle.com/datasets/mohamedjamyl/music-recommendation-system-using-spotify-dataset

## 1. Veri Ön İşleme

### 1.1 Gereksiz Sütunların Çıkarılması
Veri setinde model için gerekli olmayan sütunlar (örneğin benzersiz kimlik belirteci olan id sütunu) çıkarılmıştır.

### 1.2 Tarih Formatının Düzenlenmesi
- release_date sütunu, tarih formatına dönüştürülmüş ve yalnızca yıl bilgisi alınarak release_year sütunu oluşturulmuştur.

### 1.3 Sanatçıların Ayrıştırılması
- artists sütunundaki liste formatındaki veriler temizlenmiş ve yalnızca ilk sanatçı bilgisi alınmıştır.

### 1.4 Eksik Değerlerin Tespiti
- Veri setindeki eksik değerler kontrol edilmiş ve gerektiğinde uygun önlemler alınmıştır.

### 1.5 Özelliklerin Ölçeklenmesi
- Sayısal özellikler, MinMaxScaler kullanılarak 0 ile 1 arasında ölçeklendirilmiştir. Ölçeklenen sütunlar şunlardır:
  - valence, acousticness, danceability, duration_ms, energy, instrumentalness, liveness, loudness, speechiness, tempo

### 1.6 Korelasyon Analizi
- Korelasyon matrisi oluşturulmuş ve düşük korelasyona sahip özellikler veri setinden çıkarılmıştır.

---

## 2. Kullanılan Makine Öğrenmesi Modelleri

### 2.1 Random Forest Regressor
- **Tam model:** Ağaç sayısı (n_estimators) 100 olarak belirlenmiş ve veri setinin tamamıyla eğitilmiştir.
- **Ağaç sayısı azaltılmış model:** Eğitim sürecini hızlandırmak için ağaç sayısı 10 olarak belirlenmiştir.
  ```python
  rf_model_reduced = RandomForestRegressor(n_estimators=10, random_state=42)
  rf_model_reduced.fit(X_train, y_train)
  rf_pred_reduced = rf_model_reduced.predict(X_test)
  ```
- **Küçük veri alt kümesiyle model:** Eğitim verisinin yalnızca %10’u kullanılarak model oluşturulmuştur.
  ```python
  X_train_small = X_train.sample(frac=0.1, random_state=42)
  rf_model_small = RandomForestRegressor(random_state=42)
  rf_model_small.fit(X_train_small, y_train_small)
  rf_pred_small = rf_model_small.predict(X_test)
  ```
- Random Forest, karar ağaçlarının birleşimidir ve güçlü bir tahmin gücü sağlayan bir modeldir.

### 2.2 Linear Regression
- Doğrusal bir model olup, bağımsız değişkenlerin ağırlıklı toplamına dayalıdır.
- Basit, hızlı ve kolay uygulanabilir bir yöntemdir.

### 2.3 K-Nearest Neighbors (KNN) Regressor
- Verilerin birbirine olan uzaklıklarına dayanarak tahmin yapar.
- K değeri olarak 5 seçilmiştir.
- Daha düşük karmaşıklıkla iyi sonuçlar verebilir.

---

## 3. Kullanılan Hata Analizi Yöntemleri

### 3.1 Ortalama Kare Hatası (MSE)
- Tahmin edilen değerler ile gerçek değerler arasındaki farkların karelerinin ortalamasıdır.
- Küçük bir MSE, modelin iyi bir tahmin gücü olduğunu gösterir.

### 3.2 Kare Kök Ortalama Hata (RMSE)
- MSE’nin karekökü alınarak hesaplanır.
- Hatanın birimi, tahmin edilen hedef değişkenin birimine uygun hale getirilir.

### 3.3 Ortalama Mutlak Hata (MAE)
- Tahmin edilen değerler ile gerçek değerler arasındaki farkların mutlak değerlerinin ortalamasıdır.
- Hatanın büyüklüğünü anlamak için basit bir ölçüttür.

### 3.4 R-Kare (R²)
- Modelin açıklayıcı gücünü gösterir.
- 1’e ne kadar yakınsa modelin performansı o kadar iyidir.

---

## Sonuç Analizi ve Model Seçimi

Modellerin performansı, MSE, RMSE, MAE ve R² gibi hata metrikleri üzerinden değerlendirilmiştir. 

---

### Model Performansı Karşılaştırması

| Model                         | MSE        | RMSE      | MAE       | R²       |
|-------------------------------|----------- |-----------|-----------|----------|
| Random Forest (Reduced Trees) | 212.05     | 14.56     | 10.39     | 0.56     |
| Random Forest (Small Data)    | 207.22     | 14.39     | 10.99     | 0.57     |
| Linear Regression             | 265.95     | 16.31     | 13.11     | 0.44     |
| KNN                           | 241.60     | 15.54     | 11.76     | 0.50    |

#### Analiz Sonuçları

1. **Random Forest (Small Data)**: En düşük MSE ve RMSE değerleri bu modelde elde edilmiştir. R² metriği de 0.57 ile en yüksek performansa sahiptir. Ancak, veri setinin tamamı yerine sadece %10'u kullanıldığı için bu model, daha hızlı bir çalışma sağlasa da tam veri setinden faydalanılmamış olması sebebiyle uzun vadeli tahmin için sınırlıdır.

2. **Random Forest (Reduced Trees)**: Model, tam veri setiyle çalışmış ancak daha az ağaç kullanılmıştır. Performansı, Small Data modeli ile yakın değerler sergilemiştir. Bu model, daha hızlı bir çözüm sağlarken, hata metrikleri kabul edilebilir seviyede kalmıştır.

3. **Linear Regression**: MSE, RMSE ve MAE metriklerinde en yüksek hata oranlarını sergilemiştir. R² metriği ise 0.44 ile en düşük değerdir. Bu durum, doğrusal regresyonun, veri setindeki karmaşık ilişkileri yeterince yakalayamadığını göstermektedir.

4. **KNN**: Kümeleme temelli bu model, tahminler için orta seviyede bir başarı sağlamıştır. Ancak, hem hata oranları hem de hesaplama maliyetleri, Random Forest modellerine kıyasla dışında kalmıştır.

---

### Hangi Model Seçilmeli ve Neden?

**En uygun model, Random Forest (Reduced Trees) modelidir.**

1. **Performans**: Model, düşük hata oranları ve yüksek R² metriği ile doğruluğunu kanıtlamıştır. MSE ve RMSE değerleri, en yakın rakibi olan Small Data modeline yakın ancak veri setinin tamamıyla çalıştığı için daha fazla bilgiye dayalı kararlar verebilmektedir.

2. **Verimlilik**: Ağaç sayısının azaltılması, modelin eğitim ve tahmin süresini belirgin şekilde azaltmıştır. Bu, hem zaman hem de kaynak tasarrufu sağlamaktadır.

3. **Esneklik**: Random Forest modeli, çoklu karar ağaçları kullanarak karmaşık veri yapılarında iyi performans gösterebilme yeteneğine sahiptir. Küçük veri alt kümesiyle çalışılan modellerde elde edilen başarı da bu yapının bir göstergesidir.

4. **Genelleşme**: Model, doğrusal olmayan ilişkileri yakalayabilme konusunda doğrusal regresyon ve KNN gibi diğer modellere kıyasla daha başarılıdır.

---

### Sonuç

Bu çalışma, müzik veri seti üzerinden popülerlik tahmininde, Random Forest modelinin doğruluğu, verimliliği ve genelleşme kabiliyeti açısından diğer modellere kıyasla daha başarılı olduğunu göstermiştir.

Eğer kaynak ve zaman kısıtlamaları ön planda ise Random Forest Reduced Trees modeli tercih edilmelidir. Daha geniş bir analiz yapılacaksa, daha fazla ağaç sayısı ve tam veri seti kullanılarak modeli geliştirmek mümkün olacaktır.

YouTube Linki: https://youtu.be/CQ7ARr8TfX4

Bu projede, müzik veri seti kullanılarak popülerlik tahmini için yapılan adımları, kullanılan modelleri ve analizleri detaylıca ele aldık. Veri ön işleme, model karşılaştırmaları ve sonuçlar hakkında bilgi almak için videomuzu izleyebilirsiniz.

Haticenur Yalman 









