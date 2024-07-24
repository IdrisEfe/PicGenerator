# Üretken Yapay Zeka
# Denetimsiz Öğrenme Türü
# GAN
# T3 AI
# CIFAR-10 Veri Seti
# !!! Epoch, Loss, Validation 

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
    model.add(Dense(n_nodes, input_shape=son_boyut))
    model.add(LeakyReLU(alpha=0.2))
    
    # Reshape
    
    model.add(Reshape((4,4,256))) # Düğüm Sayılarını Tensöre çevirme
    
    # 1. Transpose Evrişim Katmanı
    
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # 2. Transpoze Evrişim Katmanı
    
    model.add(Conv2D(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # 3. Transpoze Evrişim Katmanı
    
    model.add(Conv2D(256, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # Çıkış Katmanı
    
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    
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
    
    (x_train,_),(_,_) = load_data
    
    x=x_train.astype('float32') # !!! Arrayi float32 türüne değiştiriyor
    
    # Verileri -1 ile 1 arasına normalize etmek: 0-255 piksel
    
    x=(x-127.5)/127.5
    
    return x
    
# Çalışmay abaşladığında farklı örnek veriler alacak

# Örnek veri seçme

def ornek_veri(dataset, n_samples):
    
    ix = randint(0, dataset.shape[0], n_samples)
    x=dataset[ix]
    
    # geen verileri etiketlemek için 
    
    y=((n_samples, 1))
    
    return x,y
    

    
    

    