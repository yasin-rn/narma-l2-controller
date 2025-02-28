import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

class NarmaL2Network(nn.Module):
    """
    NARMA-L2 yaklaşımı için sinir ağı modeli.
    Bu model, sıcaklık ve PWM değerlerini kullanarak gelecekteki sıcaklık değerlerini tahmin eder.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        """
        NarmaL2Network sınıfının yapıcı metodu.
        
        Args:
            input_size (int): Giriş boyutu (geçmiş PWM ve sıcaklık değerleri)
            hidden_size (int): Gizli katman boyutu
            output_size (int): Çıkış boyutu (tahmin edilecek sıcaklık değerleri)
            num_layers (int): LSTM katman sayısı
            dropout (float): Dropout oranı
        """
        super(NarmaL2Network, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM katmanı
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Tam bağlantılı katmanlar
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        """
        İleri yayılım.
        
        Args:
            x (torch.Tensor): Giriş tensörü [batch_size, input_size]
            
        Returns:
            torch.Tensor: Çıkış tensörü [batch_size, output_size]
        """
        # Giriş boyutunu yeniden şekillendir [batch_size, input_size] -> [batch_size, 1, input_size]
        x = x.unsqueeze(1)
        
        # LSTM katmanı
        lstm_out, _ = self.lstm(x)
        
        # Son zaman adımının çıkışını al
        lstm_out = lstm_out[:, -1, :]
        
        # Tam bağlantılı katmanlar
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class NarmaL2Trainer:
    """
    NARMA-L2 sinir ağı modelini eğiten sınıf.
    """
    
    def __init__(self, model, device=None):
        """
        NarmaL2Trainer sınıfının yapıcı metodu.
        
        Args:
            model (NarmaL2Network): Eğitilecek model
            device (torch.device, optional): Eğitim için kullanılacak cihaz (CPU/GPU)
        """
        self.model = model
        
        # Cihazı belirle (GPU varsa GPU, yoksa CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Eğitim cihazı: {self.device}")
        
        # Modeli cihaza taşı
        self.model.to(self.device)
        
        # Kayıp fonksiyonu ve optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = None
        
        # Eğitim geçmişi
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader, num_epochs=100, learning_rate=0.001, weight_decay=1e-5, patience=10):
        """
        Modeli eğitir.
        
        Args:
            train_loader (DataLoader): Eğitim veri yükleyicisi
            val_loader (DataLoader): Doğrulama veri yükleyicisi
            num_epochs (int): Epoch sayısı
            learning_rate (float): Öğrenme oranı
            weight_decay (float): Ağırlık azaltma katsayısı
            patience (int): Erken durdurma için sabır değeri
            
        Returns:
            tuple: Eğitim ve doğrulama kayıpları
        """
        # Optimizer'ı ayarla
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Erken durdurma için değişkenler
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # Eğitim geçmişini sıfırla
        self.train_losses = []
        self.val_losses = []
        
        # Eğitim başlangıç zamanı
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Eğitim modu
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # Verileri cihaza taşı
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Gradyanları sıfırla
                self.optimizer.zero_grad()
                
                # İleri yayılım
                outputs = self.model(inputs)
                
                # Kayıp hesapla
                loss = self.criterion(outputs, targets)
                
                # Geri yayılım
                loss.backward()
                
                # Ağırlıkları güncelle
                self.optimizer.step()
                
                # Toplam kaybı güncelle
                train_loss += loss.item() * inputs.size(0)
            
            # Epoch başına ortalama eğitim kaybı
            train_loss = train_loss / len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # Doğrulama modu
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    # Verileri cihaza taşı
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # İleri yayılım
                    outputs = self.model(inputs)
                    
                    # Kayıp hesapla
                    loss = self.criterion(outputs, targets)
                    
                    # Toplam kaybı güncelle
                    val_loss += loss.item() * inputs.size(0)
            
            # Epoch başına ortalama doğrulama kaybı
            val_loss = val_loss / len(val_loader.dataset)
            self.val_losses.append(val_loss)
            
            # Her 10 epoch'ta ilerlemeyi yazdır
            if (epoch + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Eğitim Kaybı: {train_loss:.6f}, "
                      f"Doğrulama Kaybı: {val_loss:.6f}, "
                      f"Geçen Süre: {elapsed_time:.2f} saniye")
            
            # Erken durdurma kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # En iyi modeli kaydet
                torch.save(self.model.state_dict(), './data/best_model.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Erken durdurma: Epoch {epoch+1}")
                    break
        
        # Toplam eğitim süresi
        total_time = time.time() - start_time
        print(f"Eğitim tamamlandı. Toplam süre: {total_time:.2f} saniye")
        
        # En iyi modeli yükle
        self.model.load_state_dict(torch.load('./data/best_model.pth'))
        
        return self.train_losses, self.val_losses
    
    def evaluate(self, test_loader):
        """
        Modeli değerlendirir.
        
        Args:
            test_loader (DataLoader): Test veri yükleyicisi
            
        Returns:
            tuple: Gerçek ve tahmin edilen değerler, test kaybı
        """
        # Değerlendirme modu
        self.model.eval()
        
        # Tahminleri ve gerçek değerleri saklamak için listeler
        y_true = []
        y_pred = []
        
        # Test kaybı
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Verileri cihaza taşı
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # İleri yayılım
                outputs = self.model(inputs)
                
                # Kayıp hesapla
                loss = self.criterion(outputs, targets)
                
                # Toplam kaybı güncelle
                test_loss += loss.item() * inputs.size(0)
                
                # Tahminleri ve gerçek değerleri CPU'ya taşı ve listeye ekle
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(outputs.cpu().numpy())
        
        # Ortalama test kaybı
        test_loss = test_loss / len(test_loader.dataset)
        
        # NumPy dizilerine dönüştür
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        print(f"Test Kaybı: {test_loss:.6f}")
        
        return y_true, y_pred, test_loss
    
    def predict(self, inputs):
        """
        Verilen giriş için tahmin yapar.
        
        Args:
            inputs (torch.Tensor): Giriş tensörü
            
        Returns:
            torch.Tensor: Tahmin edilen çıkış tensörü
        """
        # Değerlendirme modu
        self.model.eval()
        
        # Girişi cihaza taşı
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            # İleri yayılım
            outputs = self.model(inputs)
        
        # Çıkışı CPU'ya taşı
        outputs = outputs.cpu()
        
        return outputs
    
    def save_model(self, path='./data/narma_l2_model.pth'):
        """
        Modeli kaydeder.
        
        Args:
            path (str): Modelin kaydedileceği yol
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'output_size': self.model.output_size,
            'num_layers': self.model.num_layers
        }, path)
        
        print(f"Model kaydedildi: {path}")
    
    @staticmethod
    def load_model(path='./data/narma_l2_model.pth', device=None):
        """
        Kaydedilmiş modeli yükler.
        
        Args:
            path (str): Modelin yükleneceği yol
            device (torch.device, optional): Modelin yükleneceği cihaz
            
        Returns:
            tuple: Yüklenen model ve trainer nesnesi
        """
        # Cihazı belirle
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modeli yükle
        checkpoint = torch.load(path, map_location=device)
        
        # Modeli oluştur
        model = NarmaL2Network(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size'],
            num_layers=checkpoint['num_layers']
        )
        
        # Model durumunu yükle
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Trainer oluştur
        trainer = NarmaL2Trainer(model, device)
        
        print(f"Model yüklendi: {path}")
        
        return model, trainer
