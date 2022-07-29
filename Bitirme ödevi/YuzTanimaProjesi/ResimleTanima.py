#Gerekli kütüphanelerimiz
import cv2
import face_recognition

#İlk önce görüntülerimizi alacağız ve RGB'ye dönüştüreceğiz. Çünkü kütüphane, görüntüleri RGB olarak anlayabiliyor

imgMario = face_recognition.load_image_file("EgitimResimleri/Mario Gomez.jpg")
imgMario = cv2.cvtColor(imgMario, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("EgitimResimleri/Neymar JR.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


#Görüntünün yüz tespiti
yuzKonum = face_recognition.face_locations(imgMario)[0]
marioKodlama = face_recognition.face_encodings(imgMario)[0]
cv2.rectangle(imgMario, (yuzKonum[3],yuzKonum[0]),(yuzKonum[1],yuzKonum[2]),(255,0,255),2)

#Test görüntümüzün yüz tespiti
testKonumKodlama = face_recognition.face_locations(imgTest)[0]
testKodlama = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (testKonumKodlama[3],testKonumKodlama[0]),(testKonumKodlama[1],testKonumKodlama[2]),(255,0,255),2)

#iki resmin karşılaştırılması
sonuclar = face_recognition.compare_faces([marioKodlama],testKodlama)
yuzlerArasiMesafe = face_recognition.face_distance([marioKodlama],testKodlama)
print(sonuclar,yuzlerArasiMesafe)

#Görüntüye text ekleme
cv2.putText(imgTest,f'{sonuclar} {round(yuzlerArasiMesafe[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



cv2.imshow('Mario Gomez', imgMario) #Karşılaştırılacak görüntü
cv2.imshow('Neymar JR', imgTest) #Test görüntüsü
cv2.waitKey(0) #0 gecikme verdik


