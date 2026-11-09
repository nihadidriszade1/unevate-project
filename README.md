# Proof of Future: canlı demo və sübut paketi

Müştəri getməzdən əvvəl getmə səbəbini anlayan və kupon əvəzinə həmin səbəbi aradan qaldıran retention SDK-sının hakaton demosu.

## Paketin tərkibi

| Qovluq | Nə var |
|---|---|
| `demo/` | Canlı demo: `index.html` və 6 məhsul şəkli. İnternetsiz də işləyir. |
| `ml/` | Alış niyyəti modeli: öyrətmə skripti (`pof_intent_model.py`) və çəkilər (`pof_model.json`) |
| `evidence/` | Əsas fərziyyənin sübutu: analiz skripti, nəticələr (`evidence.json`), slayd qrafiki (`evidence_chart.png`) |
| `video/` | Ehtiyat demo videoları, 1080p, təxminən 75 saniyə: Azərbaycan və ingilis dilində izah yazıları ilə |
| `get_data.sh` | Analizlər üçün iki açıq dataseti yükləyir |

## Demonu işə salmaq

`demo/index.html` faylını Chrome-da açın. Server lazım deyil.

- İnternet yoxdursa, Google şriftləri yüklənmir və sistem şriftləri istifadə olunur. Hər şey işləyir.
- Onlayn versiya: https://claude.ai/artifact/XgKPTTaD2gDzkQDoeVUwiu
- Ən yaxşı görünüş: 1366×768 və ya daha böyük ekran, brauzer tam ekran rejimində (F11).

### Klaviatura qısayolları

| Düymə | Nə edir |
|---|---|
| `1` `2` `3` `4` | Ssenarini başladır: qiymət, ödəniş, ölçü, gecikmə |
| `R` | Cari ssenarini yenidən başladır |
| `P` | Pitch və Ətraflı rejim arasında keçid |
| `L` | Dil: AZ → TR → EN |
| `E` | ROI hesablayıcısı: 100 riskli müştəri |
| `M` | Mənbələr: hər rəqəmin mənbəyi və eksperiment sübutu |
| `Esc` | Açıq paneli bağlayır |

### Ssenarilər

| # | Münsif nə edir | Sistem nə edir |
|---|---|---|
| 1 | Qiymətə 3 dəfə toxunur | Qiymət tərəddüdü → qiymət qalxanı |
| 1 + «Qənaətcil» profil | Eyni 3 toxunuş | «Normal davranış»: müdaxilə yoxdur |
| 2 | «Ödə» düyməsini tez-tez basır | Texniki sürtünmə → bir toxunuşla ödəniş |
| 3 | Ölçü cədvəlini açır, ölçünü 2 dəfə dəyişir | Ölçü şübhəsi → ölçü köməkçisi |
| 4 | Heç nə | Gecikmiş sifariş → şikayətdən əvvəl üzr və kompensasiya |

3-cü ssenaridə ölçü düymələri ekranın aşağısında qala bilər. Telefon ekranını bir az aşağı sürüşdürün.

### Tövsiyə olunan pitch axını (təxminən 2 dəqiqə)

1. `1`: münsif qiymətə 3 dəfə toxunur. Qərar milisaniyələr içində çıxır, pul kartında xərc və xalis qazanc görünür.
2. Profili «Qənaətcil»ə keçirin: eyni davranış «normal» sayılır.
3. `2`: münsif «Ödə» düyməsini əsəbi basır.
4. `M`: «Bunu fərz etmirik, ölçmüşük»: 64 000 müştərilik eksperiment.

## Əsas rəqəmlər və mənbələri

### ML modeli (alış niyyəti)

- **Data:** UCI Online Shoppers Purchasing Intention, 12 330 real e-ticarət sessiyası (CC BY 4.0).
- **Nəticə:** görülməmiş 2 466 sessiyada AUC **0.889** (gradient boosting ilə 0.903).
- **Təsir:** ən yüksək 10% sessiyada alış 70%-dir, orta göstərici 15.5%.
- **Təkrarlamaq:** `python ml/pof_intent_model.py data/online_shoppers_intention.csv`

### Sübut: səbəbə uyğun tədbir

- **Data:** Hillstrom MineThatData e-poçt eksperimenti. 64 000 müştəri təsadüfi olaraq 3 qrupa bölünüb.
- **Tədbirin orta təsiri:** alış **+87%** (95% interval: +62% … +111%).
- **Eyni tədbir, fərqli müştəri:** qadın kataloqu qadın alıcılarda **+0.51 faiz bəndi**, kişi alıcılarda +0.06 faiz bəndi (statistik cəhətdən sıfır).
- **1000 müştəriyə xalis marja** (marja 25%, alış edənə 10% kupon):
  - hamıya yanlış tədbir: **−1 $**
  - səbəbə uyğun tədbir: **+52 $**. Sıfırdan yuxarı intervalı olan yeganə siyasət budur.
- **Təkrarlamaq:** `python evidence/uplift_evidence.py data/hillstrom.csv`

### Hələ fərziyyə olanlar (pilotda ölçüləcək)

- **Fərziyyələr:** səbəb həddi (0.62), siqnalların yarımömrü, etibar dəyişiklikləri, qalxanın xərci, marja və 90 günlük müştəri dəyəri, tərəddüd zonasındakı müştərilərin payı (60%).
- **Hamısı demoda görünür:** Mənbələr panelində hər birinin statusu və pilotda necə ölçüləcəyi yazılıb.
- **Dürüst məhdudiyyətlər:**
  - Sübut eksperimenti e-poçt kontekstindədir, tətbiq daxili təklif deyil.
  - "Hansı tədbir" sualı sübut olunub, "kimə" sualı isə hələ sübut olunmayıb.
  - Səbəb qatı hələ şəffaf qaydalarla işləyir, çünki səbəb etiketli açıq dataset yoxdur.

## Analizləri təkrarlamaq

```bash
./get_data.sh
pip install pandas scikit-learn matplotlib
python ml/pof_intent_model.py data/online_shoppers_intention.csv
python evidence/uplift_evidence.py data/hillstrom.csv
```

## Mənbələr

- UCI Online Shoppers Purchasing Intention Dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Hillstrom MineThatData E-Mail Analytics: http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- Rage click tərifi (Contentsquare): https://support.contentsquare.com/hc/en-us/articles/37271767248273-Frustration-score
- Baumeister və həmkarları (2001), Bad Is Stronger Than Good: https://fbaum.unc.edu/teaching/articles/Baumeister_2001.pdf
- Kahneman və həmkarları (1993), When More Pain Is Preferred to Less: https://www.jstor.org/stable/40062570
- Məhsul şəkilləri: Unsplash (https://unsplash.com)
