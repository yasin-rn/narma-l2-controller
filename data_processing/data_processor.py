import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TemperatureDataProcessor:
    """
    Sıcaklık verilerini işleyen ve NARMA-L2 modeli için hazırlayan sınıf.
    """
    
    def __init__(self, csv_path, input_window=10, output_window=1, sampling_time=0.2):
        """
        TemperatureDataProcessor sınıfının yapıcı metodu.
        
        Args:
            csv_path (str): CSV dosyasının yolu
            input_window (int): Giriş penceresi boyutu (geçmiş veri noktaları)
            output_window (int): Çıkış penceresi boyutu (tahmin edilecek veri noktaları)
            sampling_time (float): Örnekleme zamanı (saniye)
        """
        self.csv_path = csv_path
        self.input_window = input_window
        self.output_window = output_window
        self.sampling_time = sampling_time
        self.data = None
        self.scaler_temp = MinMaxScaler(feature_range=(0, 1))
        self.scaler_pwm = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self):
        """
        CSV dosyasından verileri yükler.
        
        Returns:
            pandas.DataFrame: Yüklenen veri seti
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            print(f"Veri seti yüklendi. Boyut: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None
    
    def preprocess_data(self):
        """
        Verileri ön işleme tabi tutar.
        
        Returns:
            tuple: İşlenmiş veri seti ve ölçekleyiciler
        """
        if self.data is None:
            print("Önce load_data() metodunu çağırın.")
            return None
        
        # Eksik değerleri kontrol et ve temizle
        if self.data.isnull().sum().sum() > 0:
            print(f"Eksik değerler bulundu: {self.data.isnull().sum()}")
            self.data = self.data.dropna()
            print(f"Eksik değerler temizlendi. Yeni boyut: {self.data.shape}")
        
        # Sütun isimlerini kontrol et
        expected_columns = ['Time', 'Temp', 'PWM']
        if not all(col in self.data.columns for col in expected_columns):
            print(f"Beklenen sütunlar bulunamadı. Mevcut sütunlar: {self.data.columns}")
            # Sütun isimlerini düzelt
            if 'Temperature' in self.data.columns and 'Temp' not in self.data.columns:
                self.data = self.data.rename(columns={'Temperature': 'Temp'})
        
        # Verileri normalize et
        temp_values = self.data['Temp'].values.reshape(-1, 1)
        pwm_values = self.data['PWM'].values.reshape(-1, 1)
        
        self.data['Temp_normalized'] = self.scaler_temp.fit_transform(temp_values)
        self.data['PWM_normalized'] = self.scaler_pwm.fit_transform(pwm_values)
        
        return self.data, self.scaler_temp, self.scaler_pwm
    
    def create_sequences(self, train_size=0.8):
        """
        Zaman serisi verilerinden eğitim ve test dizileri oluşturur.
        
        Args:
            train_size (float): Eğitim seti oranı (0-1 arası)
            
        Returns:
            tuple: Eğitim ve test veri setleri (X_train, y_train, X_test, y_test)
        """
        if 'Temp_normalized' not in self.data.columns or 'PWM_normalized' not in self.data.columns:
            print("Önce preprocess_data() metodunu çağırın.")
            return None
        
        # Giriş ve çıkış dizileri oluştur
        X, y = [], []
        
        # NARMA-L2 için giriş ve çıkış dizileri oluştur
        # Giriş: [PWM(t-input_window), ..., PWM(t-1), Temp(t-input_window), ..., Temp(t-1)]
        # Çıkış: [Temp(t), Temp(t+1), ..., Temp(t+output_window-1)]
        
        temp_data = self.data['Temp_normalized'].values
        pwm_data = self.data['PWM_normalized'].values
        
        for i in range(self.input_window, len(self.data) - self.output_window + 1):
            # Giriş dizisi: Geçmiş PWM ve sıcaklık değerleri
            pwm_seq = pwm_data[i-self.input_window:i]
            temp_seq = temp_data[i-self.input_window:i]
            input_seq = np.concatenate([pwm_seq, temp_seq])
            
            # Çıkış dizisi: Gelecek sıcaklık değerleri
            output_seq = temp_data[i:i+self.output_window]
            
            X.append(input_seq)
            y.append(output_seq)
        
        X = np.array(X)
        y = np.array(y)
        
        # Eğitim ve test setlerine ayır
        train_size = int(len(X) * train_size)
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # PyTorch tensörlerine dönüştür
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        print(f"Eğitim seti boyutu: {X_train.shape}, {y_train.shape}")
        print(f"Test seti boyutu: {X_test.shape}, {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def inverse_transform_temp(self, normalized_values):
        """
        Normalize edilmiş sıcaklık değerlerini orijinal ölçeğe dönüştürür.
        
        Args:
            normalized_values (numpy.ndarray): Normalize edilmiş sıcaklık değerleri
            
        Returns:
            numpy.ndarray: Orijinal ölçekteki sıcaklık değerleri
        """
        return self.scaler_temp.inverse_transform(normalized_values)
    
    def inverse_transform_pwm(self, normalized_values):
        """
        Normalize edilmiş PWM değerlerini orijinal ölçeğe dönüştürür.
        
        Args:
            normalized_values (numpy.ndarray): Normalize edilmiş PWM değerleri
            
        Returns:
            numpy.ndarray: Orijinal ölçekteki PWM değerleri
        """
        return self.scaler_pwm.inverse_transform(normalized_values)
    
    def plot_data(self):
        """
        Veri setini görselleştirir.
        """
        if self.data is None:
            print("Önce load_data() metodunu çağırın.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Sıcaklık grafiği
        plt.subplot(2, 1, 1)
        plt.plot(self.data['Time'], self.data['Temp'], 'b-', label='Sıcaklık')
        plt.ylabel('Sıcaklık (°C)')
        plt.title('Sıcaklık ve PWM Değişimi')
        plt.grid(True)
        plt.legend()
        
        # PWM grafiği
        plt.subplot(2, 1, 2)
        plt.plot(self.data['Time'], self.data['PWM'], 'r-', label='PWM')
        plt.xlabel('Zaman')
        plt.ylabel('PWM (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('temperature_pwm_plot.png')
        plt.close()
        
        print("Grafik kaydedildi: temperature_pwm_plot.png")


class TemperatureDataset(Dataset):
    """
    PyTorch veri seti sınıfı.
    """
    
    def __init__(self, X, y):
        """
        TemperatureDataset sınıfının yapıcı metodu.
        
        Args:
            X (torch.Tensor): Giriş verileri
            y (torch.Tensor): Çıkış verileri
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        """
        Veri seti uzunluğunu döndürür.
        
        Returns:
            int: Veri seti uzunluğu
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Belirtilen indeksteki veri örneğini döndürür.
        
        Args:
            idx (int): Veri örneği indeksi
            
        Returns:
            tuple: Giriş ve çıkış tensörleri
        """
        return self.X[idx], self.y[idx]


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    PyTorch DataLoader nesneleri oluşturur.
    
    Args:
        X_train (torch.Tensor): Eğitim giriş verileri
        y_train (torch.Tensor): Eğitim çıkış verileri
        X_test (torch.Tensor): Test giriş verileri
        y_test (torch.Tensor): Test çıkış verileri
        batch_size (int): Batch boyutu
        
    Returns:
        tuple: Eğitim ve test DataLoader nesneleri
    """
    train_dataset = TemperatureDataset(X_train, y_train)
    test_dataset = TemperatureDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
