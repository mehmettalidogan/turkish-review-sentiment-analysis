# Veri seti temizleme ve sonuç işlemleri

Bu not, Trendyol yorumlarından duygu analizi çalışma setine kadar yapılan hazırlığı özetler. Amaç, ham dosyayı bozmadan üç sınıflı ve dengeli bir küme üretmek, ardından aynı yorumun hem eğitimde hem testte görünmesini engellemektir.

İşlem kodu dört betikte durur:

| Betik | Ne yapar | Çıktı dosyası |
| --- | --- | --- |
| `analyze_dataset.py` | Boyut, sütun, eksik değer, tam satır tekrarı ve olası metin sütununu raporlar | Yok (yalnızca ekran) |
| `analyze_sentiment_mapping.py` | Yıldız başına yorum uzunluğu, çok kısa yorum ve örnek metinlere bakar | Yok (yalnızca ekran) |
| `prepare_dataset.py` | Yıldızı duygu etiketine çevirir, kontrolleri basar, sınıf başına örnek çeker | `dataset/balanced_sentiment.csv` |
| `split_train_test.py` | Dengeli kümeyi yoruma göre bölünmeden %80 / %20 ayırır | `dataset/train_sentiment.csv`, `dataset/test_sentiment.csv` |

Ham dosya `dataset/trendyol_reviews_20251212_000835.csv` hiç üzerine yazılmadı. Dengeli dosya da eğitim/test ayrımında yalnızca okundu.

## Ham veri

582.601 yorum satırı var. Tamamen aynı olan satır yok. `seller_rate` sütununda 20 eksik değer var; diğer sütunlarda eksik yok. Yorum metninde `null` veya yalnızca boşluktan oluşan satır yok.

Sütunlar: `category`, `product_url`, `user`, `date`, `point`, `comment`, `seller`, `image_count`, `seller_rate`.

`point` 1–5 yıldızdır. `date` Türkçe düz metindir (`24 Eylül 2024`). Kullanıcı adları maskelenmiş gelir (`S** K**`). Kategoriler:

| Kategori | Satır |
| --- | ---: |
| Aksesuar | 169.878 |
| Giyim | 154.784 |
| Kozmetik & Kişisel Bakım | 97.098 |
| Elektronik | 85.759 |
| Kitap | 75.082 |

Aynı yorum metni birden fazla satırda geçebilir. 503.308 benzersiz `comment` vardır; 79.293 satır bu yüzden fazladan tekrardır. Tekrarlar silinmedi. Aynı cümle farklı ürün, satıcı veya kullanıcıda durabilir; bazı tekrarların yıldızı da birbirini tutmaz.

41 yorumun uzunluğu 3 karakter veya daha kısadır (3 yıldızda 2, 4 yıldızda 5, 5 yıldızda 34). Bunlar da silinmedi. Yıldız başına ortalama yorum uzunluğu kabaca 1 yıldızda 75, 2 yıldızda 69, 3–5 yıldızda 55–58 karakterdir.

## Duygu etiketi

Yıldız şu kurala göre `sentiment` sütununa yazıldı:

| Yıldız | Duygu |
| --- | --- |
| 1 ve 2 | Negative |
| 3 | Neutral |
| 4 ve 5 | Positive |

Eşleşmeyen yıldız yok. Ham dağılım çok dengesizdir:

| Duygu | Satır | Pay |
| --- | ---: | ---: |
| Positive | 558.487 | %95,9 |
| Neutral | 14.571 | %2,5 |
| Negative | 9.543 | %1,6 |

Yıldız kırılımı: 1 → 6.288, 2 → 3.255, 3 → 14.571, 4 → 61.152, 5 → 497.335.

Aynı metin birden fazla duygu sınıfında da geçebiliyor. 1.469 benzersiz yorumda etiket çatışması var; bu yorumlar toplam 31.943 satıra yayılır:

| Birlikte görülen etiketler | Benzersiz yorum |
| --- | ---: |
| Neutral + Positive | 1.187 |
| Negative + Positive | 147 |
| Negative + Neutral + Positive | 93 |
| Negative + Neutral | 42 |

Bu satırlar silinmedi ve etiketleri değiştirilmedi. Çatışma, aynı metnin farklı ürünlerde farklı yıldız almasından gelir; tek bir “doğru” sınıf seçmek veriye dışarıdan hüküm yazmak olurdu.

Metin üzerinde başka bir temizlik yapılmadı: küçük harfe çevirme, emoji silme, noktalama sadeleştirme veya yazım düzeltmesi yok. `seller_rate` boşlukları da doldurulmadı.

## Dengeli çalışma kümesi

`prepare_dataset.py` her sınıftan `random_state=42` ile 9.000 satır çeker, üç parçayı birleştirir ve aynı tohumla karıştırır. Sonuç `dataset/balanced_sentiment.csv`: 27.000 satır, üç sınıfta da 9.000.

Negative kaynakta yalnızca 9.543 satır olduğu için bu sınıfın neredeyse tamamı alındı (yaklaşık %94). Neutral’da 14.571 satırdan 9.000’i, Positive’ta 558.487 satırdan 9.000’i seçildi. Positive örneklemi kaynak oranını korur: dengeli kümede 7.996 adet 5 yıldız ve 1.004 adet 4 yıldız vardır. Negative tarafında 5.920 adet 1 yıldız ve 3.080 adet 2 yıldız vardır. Neutral’ın tamamı 3 yıldızdır.

Dengeli kümede de tekrar ve çatışma duruyor, çünkü örnekleme bunları elemez:

- 25.697 benzersiz yorum, 1.303 fazla tekrar satırı
- Tekrar eden bir metne ait 1.793 satır
- 247 yorum hâlâ birden fazla duygu etiketi taşır (1.217 satır)

Bu 247 yorumun kombinasyonları: Neutral + Positive 126, Negative + Neutral 55, üç sınıf birden 45, Negative + Positive 21.

Kategori dağılımı örneklemeden sonra: Giyim 8.545, Aksesuar 7.672, Kozmetik & Kişisel Bakım 4.400, Elektronik 3.851, Kitap 2.532. Boş yorum kalmadı.

## Eğitim ve test ayrımı

`split_train_test.py` satırı tek tek bölmez. Her benzersiz `comment` bir gruptur; grubun bütün satırları ya eğitime ya teste gider. Böylece aynı metin iki tarafta birden öğrenilip ölçülmez.

Hedef, her sınıfın %20’sidir: 9.000 × 0,20 = 1.800. Gruplar `random_state=42` ile karıştırılır. Bir grup teste eklenince herhangi bir sınıf 1.800’ü aşacaksa o grup eğitimde kalır. Bu çalıştırmada hedefler tam doldu.

| Küme | Dosya | Satır | Negative | Neutral | Positive | Benzersiz yorum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Eğitim | `dataset/train_sentiment.csv` | 21.600 | 7.200 | 7.200 | 7.200 | 20.602 |
| Test | `dataset/test_sentiment.csv` | 5.400 | 1.800 | 1.800 | 1.800 | 5.095 |

İki dosyanın yorum kesişimi 0’dır. Satır toplamı 27.000’dir; dengeli dosyadan satır kaybı yoktur. Dosyalar yazıldıktan sonra aynı kesişim kontrolü diskten tekrar okunarak doğrulandı.

Testteki yıldız kırılımı: Negative 1.190 + 610, Neutral 1.800, Positive 206 + 1.594. Eğitim bunun kalanıdır.

## Bilinçli olarak yapılmayanlar

- Tekrarlayan yorumlar silinmedi.
- Çatışan etiketler silinmedi veya tek sınıfa indirgenmedi.
- 3 karakter ve daha kısa 41 yorum atılmadı.
- Metin normalize edilmedi.
- Ham CSV ve `balanced_sentiment.csv` sonraki adımlarda değiştirilmedi.

Sonraki model adımı `comment` metnini girdi, `sentiment` sütununu hedef almalıdır. `point` hedef olarak kullanılmamalıdır; duygu etiketi zaten yıldızdan üretilmiştir ve modeli yıldıza göre değerlendirmek aynı bilgiyi iki kez sayar.
