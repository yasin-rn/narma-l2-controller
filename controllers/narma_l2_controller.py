import torch
import numpy as np
from models.neural_network import NarmaL2Trainer
import time

class NarmaL2Controller:
    """
    NARMA-L2 (Nonlinear AutoRegressive Moving Average) kontrolcüsü.
    Bu kontrolcü, eğitilmiş bir sinir ağı modelini kullanarak sıcaklık kontrolü yapar.
    """
    
    def __init__(self, model_path, data_processor, device=None):
        """
        NarmaL2Controller sınıfının yapıcı metodu.
        
        Args:
            model_path (str): Eğitilmiş modelin yolu
            data_processor (TemperatureDataProcessor): Veri işleme sınıfı
            device (torch.device, optional): Modelin çalıştırılacağı cihaz
        """
        # Modeli yükle
        self.model, self.trainer = NarmaL2Trainer.load_model(model_path, device)
        
        # Veri işleme sınıfı
        self.data_processor = data_processor
        
        # Giriş penceresi boyutu
        self.input_window = data_processor.input_window
        
        # Geçmiş değerleri saklamak için tamponlar
        self.temp_buffer = []
        self.pwm_buffer = []
        
        # Örnekleme zamanı
        self.sampling_time = data_processor.sampling_time
        
        # Kontrolcü parametreleri
        self.min_pwm = 0.0
        self.max_pwm = 100.0
        
        print("NARMA-L2 kontrolcüsü başlatıldı.")
    
    def initialize_buffers(self, initial_temp, initial_pwm):
        """
        Tamponları başlangıç değerleriyle doldurur.
        
        Args:
            initial_temp (float): Başlangıç sıcaklık değeri
            initial_pwm (float): Başlangıç PWM değeri
        """
        # Normalize et
        temp_normalized = self.data_processor.scaler_temp.transform([[initial_temp]])[0][0]
        pwm_normalized = self.data_processor.scaler_pwm.transform([[initial_pwm]])[0][0]
        
        # Tamponları doldur
        self.temp_buffer = [temp_normalized] * self.input_window
        self.pwm_buffer = [pwm_normalized] * self.input_window
        
        print(f"Tamponlar başlatıldı: Sıcaklık={initial_temp}°C, PWM={initial_pwm}%")
    
    def update_buffers(self, current_temp, current_pwm):
        """
        Tamponları günceller.
        
        Args:
            current_temp (float): Güncel sıcaklık değeri
            current_pwm (float): Güncel PWM değeri
        """
        # Normalize et
        temp_normalized = self.data_processor.scaler_temp.transform([[current_temp]])[0][0]
        pwm_normalized = self.data_processor.scaler_pwm.transform([[current_pwm]])[0][0]
        
        # Tamponları güncelle (FIFO - İlk giren ilk çıkar)
        self.temp_buffer.pop(0)
        self.temp_buffer.append(temp_normalized)
        
        self.pwm_buffer.pop(0)
        self.pwm_buffer.append(pwm_normalized)
    
    def predict_temperature(self, pwm_value):
        """
        Belirli bir PWM değeri için gelecekteki sıcaklığı tahmin eder.
        
        Args:
            pwm_value (float): PWM değeri
            
        Returns:
            float: Tahmin edilen sıcaklık değeri
        """
        # PWM değerini normalize et
        pwm_normalized = self.data_processor.scaler_pwm.transform([[pwm_value]])[0][0]
        
        # Geçici tamponları kopyala
        temp_buffer_copy = self.temp_buffer.copy()
        pwm_buffer_copy = self.pwm_buffer.copy()
        
        # Son PWM değerini güncelle
        pwm_buffer_copy.pop(0)
        pwm_buffer_copy.append(pwm_normalized)
        
        # Giriş dizisini oluştur
        input_seq = np.concatenate([pwm_buffer_copy, temp_buffer_copy])
        
        # PyTorch tensörüne dönüştür
        input_tensor = torch.FloatTensor([input_seq])
        
        # Tahmin yap
        output_tensor = self.trainer.predict(input_tensor)
        
        # Normalize edilmiş tahmini al
        predicted_temp_normalized = output_tensor.numpy()[0][0]
        
        # Orijinal ölçeğe dönüştür
        predicted_temp = self.data_processor.inverse_transform_temp([[predicted_temp_normalized]])[0][0]
        
        return predicted_temp
    
    def calculate_pwm(self, target_temp, current_temp, search_resolution=1.0):
        """
        Hedef sıcaklığa ulaşmak için gerekli PWM değerini hesaplar.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            current_temp (float): Güncel sıcaklık değeri
            search_resolution (float): Arama çözünürlüğü
            
        Returns:
            float: Hesaplanan PWM değeri
        """
        # Hedef sıcaklık ve güncel sıcaklık arasındaki fark
        temp_diff = target_temp - current_temp
        
        # Eğer fark çok küçükse, mevcut PWM değerini koru
        if abs(temp_diff) < 0.1:
            return self.pwm_buffer[-1] * 100.0  # Normalize edilmiş değeri orijinal ölçeğe dönüştür
        
        # Sistemin çalışma modunu belirle (ısıtma veya soğutma)
        # Veri setindeki ilişkiye göre, PWM değeri arttıkça sıcaklık düşüyor (soğutma modu)
        is_cooling_mode = True
        
        # Eğer hedef sıcaklık, mevcut sıcaklıktan düşükse ve soğutma modundaysak
        # veya hedef sıcaklık, mevcut sıcaklıktan yüksekse ve ısıtma modundaysak
        # doğrudan PWM değerini ayarla
        if (temp_diff < 0 and is_cooling_mode) or (temp_diff > 0 and not is_cooling_mode):
            # Sıcaklık farkına göre PWM değerini hesapla
            # Fark ne kadar büyükse, PWM değeri o kadar yüksek olmalı
            pwm_value = min(max(abs(temp_diff) * 10.0, self.min_pwm), self.max_pwm)
            return pwm_value
        else:
            # Aksi takdirde, PWM değerini minimumda tut
            return self.min_pwm
        
        # Aşağıdaki kod, tahmin modeli kullanarak PWM değerini hesaplar
        # Ancak veri setindeki ilişki nedeniyle doğru çalışmıyor
        # Bu nedenle, yukarıdaki doğrudan hesaplama yöntemini kullanıyoruz
        """
        # PWM değerlerini tara ve en iyi değeri bul
        best_pwm = None
        min_error = float('inf')
        
        for pwm in np.arange(self.min_pwm, self.max_pwm + search_resolution, search_resolution):
            # Belirli bir PWM değeri için sıcaklığı tahmin et
            predicted_temp = self.predict_temperature(pwm)
            
            # Tahmin edilen sıcaklık ile hedef sıcaklık arasındaki hatayı hesapla
            error = abs(predicted_temp - target_temp)
            
            # Eğer hata daha küçükse, en iyi PWM değerini güncelle
            if error < min_error:
                min_error = error
                best_pwm = pwm
        
        return best_pwm
        """
    
    def control_step(self, target_temp, current_temp, machine_control=None):
        """
        Bir kontrol adımı gerçekleştirir.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            current_temp (float): Güncel sıcaklık değeri
            machine_control (MachineControl, optional): Makine kontrol sınıfı
            
        Returns:
            tuple: Hesaplanan PWM değeri ve tahmin edilen sıcaklık
        """
        # Son PWM değerini al (normalize edilmiş değeri orijinal ölçeğe dönüştür)
        last_pwm = self.data_processor.inverse_transform_pwm([[self.pwm_buffer[-1]]])[0][0]
        
        # Hedef sıcaklığa ulaşmak için gerekli PWM değerini hesapla
        calculated_pwm = self.calculate_pwm(target_temp, current_temp)
        
        # Hesaplanan PWM değeri için tahmin edilen sıcaklık
        predicted_temp = self.predict_temperature(calculated_pwm)
        
        # Eğer makine kontrol sınıfı verilmişse, PWM değerini ayarla
        if machine_control is not None:
            machine_control.set_pwm(calculated_pwm)
        
        # Tamponları güncelle
        self.update_buffers(current_temp, calculated_pwm)
        
        return calculated_pwm, predicted_temp
    
    def control_loop(self, target_temp, duration, machine_control):
        """
        Belirli bir süre boyunca kontrol döngüsü çalıştırır.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            duration (float): Kontrol döngüsü süresi (saniye)
            machine_control (MachineControl): Makine kontrol sınıfı
            
        Returns:
            tuple: Zaman, sıcaklık ve PWM değerlerini içeren diziler
        """
        # Sonuçları saklamak için diziler
        time_values = []
        temp_values = []
        pwm_values = []
        
        # Başlangıç zamanı
        start_time = time.time()
        elapsed_time = 0.0
        
        print(f"Kontrol döngüsü başlatıldı. Hedef sıcaklık: {target_temp}°C, Süre: {duration} saniye")
        
        while elapsed_time < duration:
            # Güncel sıcaklığı oku
            current_temp = machine_control.read_temperature()
            
            if current_temp is None:
                print("Sıcaklık okunamadı. Kontrol döngüsü durduruldu.")
                break
            
            # Kontrol adımını gerçekleştir
            calculated_pwm, predicted_temp = self.control_step(target_temp, current_temp, machine_control)
            
            # Sonuçları kaydet
            time_values.append(elapsed_time)
            temp_values.append(current_temp)
            pwm_values.append(calculated_pwm)
            
            # Her 5 saniyede bir ilerlemeyi yazdır
            if int(elapsed_time) % 5 == 0 and int(elapsed_time) > 0:
                print(f"Zaman: {elapsed_time:.1f}s, Sıcaklık: {current_temp:.2f}°C, PWM: {calculated_pwm:.2f}%, "
                      f"Tahmin: {predicted_temp:.2f}°C")
            
            # Örnekleme zamanı kadar bekle
            time.sleep(self.sampling_time)
            
            # Geçen süreyi güncelle
            elapsed_time = time.time() - start_time
        
        print(f"Kontrol döngüsü tamamlandı. Toplam süre: {elapsed_time:.1f} saniye")
        
        return np.array(time_values), np.array(temp_values), np.array(pwm_values)
    
    def simulate_control(self, target_temp, initial_temp, duration, time_step=0.2):
        """
        Kontrol sistemini simüle eder.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            initial_temp (float): Başlangıç sıcaklık değeri
            duration (float): Simülasyon süresi (saniye)
            time_step (float): Zaman adımı (saniye)
            
        Returns:
            tuple: Zaman, sıcaklık ve PWM değerlerini içeren diziler
        """
        # Tamponları başlat
        self.initialize_buffers(initial_temp, 0.0)
        
        # Sonuçları saklamak için diziler
        time_values = []
        temp_values = []
        pwm_values = []
        
        # Simülasyon başlangıcı
        current_time = 0.0
        current_temp = initial_temp
        
        print(f"Simülasyon başlatıldı. Hedef sıcaklık: {target_temp}°C, Başlangıç sıcaklığı: {initial_temp}°C")
        
        while current_time < duration:
            # Kontrol adımını gerçekleştir
            calculated_pwm, predicted_temp = self.control_step(target_temp, current_temp)
            
            # Sonuçları kaydet
            time_values.append(current_time)
            temp_values.append(current_temp)
            pwm_values.append(calculated_pwm)
            
            # Her 5 saniyede bir ilerlemeyi yazdır
            if int(current_time) % 5 == 0 and int(current_time) > 0:
                print(f"Zaman: {current_time:.1f}s, Sıcaklık: {current_temp:.2f}°C, PWM: {calculated_pwm:.2f}%, "
                      f"Tahmin: {predicted_temp:.2f}°C")
            
            # Bir sonraki zaman adımı için sıcaklığı güncelle (basit bir model)
            # Gerçek sistemde bu değer, makine kontrol sınıfından okunacak
            current_temp = predicted_temp
            
            # Zaman adımını güncelle
            current_time += time_step
        
        print(f"Simülasyon tamamlandı. Toplam süre: {duration:.1f} saniye")
        
        return np.array(time_values), np.array(temp_values), np.array(pwm_values)
