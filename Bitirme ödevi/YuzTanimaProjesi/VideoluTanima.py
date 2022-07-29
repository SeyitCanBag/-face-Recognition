#Gerekli kütüphanelerimiz
import cv2
import numpy as np
import face_recognition
import os

#İlk aşama resimlerimizi içe aktarmak
resimUzantisi = "VideoEgitimResimleri"
resimler = []
resimIsimleri = []
goruntuListesi = os.listdir(resimUzantisi)
print(goruntuListesi)

#Bu kısımda yine resimlerin listesini uzantıları olmadan alıyoruz.
for cl in goruntuListesi: #Tüm görüntüleri alıyoruz
    topResimler = cv2.imread(f'{resimUzantisi}/{cl}') #tüm görüntüleri teker teker alıyoruz.
    resimler.append(topResimler) #ve görüntüyü resimler listesine ekliyoruz.
    resimIsimleri.append(os.path.splitext(cl)[0]) #yine görüntülerin isimlerini aldık ama .jpg kısımlarını almadık.
print(resimIsimleri)

#Bu aşamada resimleri kodlamak istediğimizi söylüyoruz
def kodla(resimler):
    kodlanmakIstenenResimlerListesi = []
    for resim in resimler:
        resim = cv2.cvtColor(resim,cv2.COLOR_BGR2RGB)
        resmiKodla = face_recognition.face_encodings(resim)[0]
        kodlanmakIstenenResimlerListesi.append(resmiKodla)
    return kodlanmakIstenenResimlerListesi


kodlanmisResimler = kodla(resimler)
print(len(kodlanmisResimler))




#Bu adımda bizim ve kodlamalarımız arasındaki eşleşmeyi bulmak olacak
kamera = cv2.VideoCapture(0)
while True:
    basari, resim = kamera.read()
    iyilestirilmisResim = cv2.resize(resim,(0,0),None,0.25,0.25)
    iyilestirilmisResim = cv2.cvtColor(iyilestirilmisResim, cv2.COLOR_BGR2RGB)

    #şimdi ise yüz konumlarını bulacağız ve onu kodlamak istediğimizi söylüyoruz
    yuzunKonumu = face_recognition.face_locations(iyilestirilmisResim)
    iyilestirilmisiKodla = face_recognition.face_encodings(iyilestirilmisResim,yuzunKonumu)

    #Yüz görüntülerini karşılaştırdığımız kısım
    for kodlananYuz,konum in zip(iyilestirilmisiKodla, yuzunKonumu):
        karsilastirma = face_recognition.compare_faces(kodlanmisResimler,kodlananYuz)
        mesafe = face_recognition.face_distance(kodlanmisResimler, kodlananYuz)
        print(mesafe)
        minMesafe = np.argmin(mesafe)

        #Şimdi sınırlayıcı bir kutu çizebiliriz ve isimlerini yazdırabiliriz
        if karsilastirma[minMesafe]:
            isim = resimIsimleri[minMesafe].upper()
            print(isim)

            # Şimdi sınırlayıcı bir kutu ve adını yazdıracağız.
            y1, x2, y2, x1 = konum
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(resim, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(resim, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(resim, isim, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Kamera', resim)
    cv2.waitKey(1)
