# Üretken Yapay Zeka
# Denetimsiz Öğrenme Türü
# GAN
# T3 AI
# CIFAR-10 Veri Seti
# !!! Epoch, Loss, Validation, Batch

'''
üretici çekişmeli ağlar ile görüntü oluşturma
'''

# gerekli kütüphaneler
from numpy import zeros, ones # Sahte, gerçek ayrımı
from numpy.random import randint, randn # Rastgele fotoğraf seçme
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam # !!!
import matplotlib.pyplot as plt

# ayrımcı model tanımlayalım

def ayrimci(in_shape=(32,32,3)):
    
    model = Sequential()
    
    # 1. Evrişim
    model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape)) # !!!
    model.add(LeakyReLU(alpha=0.2)) # 0 ın altı verilerin %20 si
    
    # 2. Evrişim
    
    model.add(Conv2D(128, (3,3), strides=(2,2), # 2x2 piksel değiştir
                     padding='same')) # Giriş ve Çıkış verileri aynı boyutta olsun diye
    model.add(LeakyReLU(alpha=0.2))
    
    # 3. Evrişim
    
    model.add(Conv2D(128, (3,3), strides=(2,2),
                     padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # 4. Evrişim 
    
    model.add(Conv2D(256, (3,3), strides=(2,2), 
                     padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # Flatten()
    
    model.add(Flatten())
    
    # Dropout Katmanı
    
    model.add(Dropout(0.4))
    
    # Çıkış Katmanı
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizer algoritması
    
    opt=Adam(learning_rate=0.0002, beta_1=0.5) # !!! beta_1
    
    # Model Değerlendirme
    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    
    return model


# Üretici model oluşturma

def uretici(son_boyut):
    
    model=Sequential()
    
    # 1. Dense Katmanı
    
    n_nodes = 256*4*4 # 256 katman 4x4
    model.add(Dense(n_nodes, input_dim=son_boyut))
    model.add(LeakyReLU(alpha=0.2))
    
    # Reshape
    
    model.add(Reshape((4,4,256))) # Düğüm Sayılarını Tensöre çevirme
    
    # 1. Transpose Evrişim Katmanı
    
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # 2. Transpoze Evrişim Katmanı
    
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # 3. Transpoze Evrişim Katmanı
    
    model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # Çıkış Katmanı
    
    model.add(Conv2DTranspose(3, (3,3), activation='tanh', padding='same'))
    
    return model

# Üretici ve AYrıştırıcı ayrıştırma

def Gans(g_model, d_model):
    
    d_model.trainable=False
    
    model=Sequential()
    
    model.add(g_model)
    model.add(d_model)
    
    opt=Adam(learning_rate=0.0002, beta_1 = 0.5)
    
    model.compile(optimizer=opt, loss='binary_crossentropy')
    
    return model

# Veri Seti Yükleme

def veri_seti():
    
    (x_train,_),(_,_) = load_data()
    
    x=x_train.astype('float32') # !!! Arrayi float32 türüne değiştiriyor
    
    # Verileri -1 ile 1 arasına normalize etmek: 0-255 piksel
    
    x=(x-127.5)/127.5
    
    return x
    
# Çalışmay abaşladığında farklı örnek veriler alacak

# Örnek veri seçme

def ornek_veri(dataset, n_samples):
    
    ix = randint(0, dataset.shape[0], n_samples)
    x=dataset[ix]
    
    # gelen verileri etiketlemek için 
    
    y=ones((n_samples, 1))
    
    return x,y

# gizli noktalar

def nokta_olusturma(son_boyut, n_samples):
    
    x_input = randn(son_boyut*n_samples)
    x_input = x_input.reshape(n_samples, son_boyut) # Ters çevirdik
    
    return x_input
    
def sahte_nesne_olusturma(g_model, son_boyut, n_samples):
    
    x_input = nokta_olusturma(son_boyut, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples,1))
    
    return X,y

# cizim

def cizim(examples, epoch, n=7):
    
    # n ızgara sayısı
    
    examples = (examples+1)/2.0
    
    for i in range(n*n):
        plt.subplot(n,n, i+1)
        plt.axis('off')
        plt.imshow(examples[i])
        
    filename = 'olusturulan_ornek %03d.png'%(epoch+1)
    plt.savefig(filename)
    plt.close()
    
# model değerlenidrme

def degerlendirme(epoch, g_model, d_model, dataset, son_boyut, n_samples = 150):
    
    x_gercek, y_gercek = ornek_veri(dataset, n_samples)
    loss_gercek, acc_gercek = d_model.evaluate(x_gercek, y_gercek)
    
    x_sahte, y_sahte = sahte_nesne_olusturma(g_model, son_boyut, n_samples)
    loss_sahte, acc_sahte = d_model.evaluate(x_sahte, y_sahte)
    
    print(f'Gerçek Loss ve Doğruluk değeri: ({loss_gercek},{acc_gercek}), Sahte Loss ve Doğruluk değeri: ({loss_sahte},{acc_sahte}) ')
    
    cizim(x_sahte, epoch)
    filename='olusturulan_ornek %03d.h3'%(epoch+1)
    g_model.save(filename)
    
def train(g_model, d_model, gan_model, dataset, son_boyut, n_epochs=30, n_batch=128):
    
    bat_per_epo = int(dataset.shape[0]/n_batch)
    
    half_batch = int(n_batch/2)
    
    for i in range(n_epochs):
        
        for j in range(bat_per_epo):
            
            x_gercek, y_gercek = ornek_veri(dataset, half_batch)
            d_loss1 = d_model.train_on_batch(x_gercek, y_gercek)
            
            x_sahte, y_sahte = sahte_nesne_olusturma(g_model, son_boyut, half_batch)
            d_loss2 = d_model.train_on_batch(x_sahte, y_sahte)
            
            x_gan = nokta_olusturma(son_boyut, n_batch)
            y_gan = ones((n_batch,1))
            g_loss = g_model.train_on_batch(x_gan, y_gan)
            
            print(f'Epoch: {i+1},'
                  f'Batch: {j+1},'
                  f"Toplam Batch:{bat_per_epo},"
                  f'Ayrımcı Modelin Gerçek Kayıp Fonksiyonu: {d_loss1[1]},'
                  f'Ayrımcı Modelin Sahte Kayıp Fonksiyonu: {d_loss2[1]},'
                  f'Üretici Modelin Kayıp Fonksiyonu: {g_loss[1]},'
                  )
            
# parametre göndermek

son_boyut=100
d_model = ayrimci()
g_model = uretici(son_boyut)
gan_model = Gans(g_model, d_model)
dataset=veri_seti()
train(g_model, d_model, gan_model, dataset, son_boyut)            
    
    
        
    
    
    