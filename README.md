# Derin Öğrenme Yaklaşımı ile Yüz Tanıma Sistemi

Bu proje, gömme temelli modern bir yüz tanıma sisteminin PyTorch ile sıfırdan geliştirilmesini kapsamaktadır. Projede, yüz algılama için MTCNN, yüz temsili üretimi için FaceNet mimarisi ve temsil uzayı optimizasyonu için Triplet Loss & Center Loss fonksiyonları birlikte kullanılmıştır. Sistem; maske, gözlük, poz değişimi gibi çevresel varyasyonlara karşı dayanıklı olacak şekilde eğitilmiş ve değerlendirilmiştir.

## Proje Amacı

Gömme temelli yüz tanıma yaklaşımlarını derinlemesine inceleyerek, yüksek doğrulukta çalışan ve gerçek zamanlı uygulanabilirliği olan bir sistem geliştirmek.

## Kullanılan Yöntemler ve Mimariler

- **Yüz Algılama:** MTCNN (Multi-task Cascaded Convolutional Networks)
- **Embedding Üretimi:** Sıfırdan tasarlanmış FaceNet mimarisi (custom CNN)
- **Kayıp Fonksiyonları:**
  - Triplet Loss
  - Center Loss (opsiyonel olarak entegre edildi)
- **Metrikler:** Öklidyen mesafesi ve Kosinüs benzerliği ile kimlik karşılaştırması

## Veri Seti

- Kullanılan veri seti: CelebA'dan özel olarak türetilmiş ve filtrelenmiş alt küme
- Kapsam:
  - 3600 birey (1800 kadın, 1800 erkek)
  - Her birey için 10 farklı poz ve varyasyona sahip görüntü
- Görüntü ön işleme:
  - MTCNN ile yüz kırpma ve hizalama
  - Normalize ([-1, 1] aralığı)
  - Veri artırma: yatay çevirme, ColorJitter, RandomAffine, RandomErasing

## Model Mimarisi

- Derinlik: 4 konvolüsyon bloğu + ortalama havuzlama + tam bağlantılı katman
- Aktivasyon: ReLU
- Düzenleme: Dropout, BatchNorm
- Optimizasyon: AdamW
- Scheduler: CosineAnnealingLR
- Triplet Mining: Semi-Hard ve Hard Negative Mining destekli

## Gerçek Zamanlı Uygulama

- OpenCV ile kamera üzerinden yüz tanıma demo uygulaması geliştirildi
- Kişi ekleme ve tanıma için iki butonlu GUI arayüzü (Tkinter)

## Altyapı

- Google Colab (Tesla T4 GPU)
- PyTorch, TorchVision, NumPy, Pandas, OpenCV, scikit-learn
- Python 3.10+

## Klasör Yapısı
- preprocessing.py/ # Ön işleme kodu 
- train.py # Model eğitim kodu
- folder.py # Dosyaların veri miktarını düzenleme kodu
- demo.py # Gerçek zamanlı demo arayüz
- README.md # Proje açıklaması

  
**Beyza Nur KILIÇ**  
Uludağ Üniversitesi - Bilgisayar Mühendisliği  
Bitirme Projesi (2025)  
Danışman: Prof. Dr. Kemal Fidanboylu
