from machine_control.machine_control import MachineControl
from data_processing.data_processor import TemperatureDataProcessor
from controllers.narma_l2_controller import NarmaL2Controller
from data_processing.visualize import plot_control_performance
import numpy as np
import time
import matplotlib.pyplot as plt

class TemperatureControl:
    """
    Sıcaklık kontrol sınıfı.
    Bu sınıf, NARMA-L2 kontrolcüsünü kullanarak sıcaklık kontrolü yapar.
    """
    
    def __init__(self, model_path, csv_path, host="192.168.1.40", port=4840, sampling_time=0.2):
        """
        TemperatureControl sınıfının yapıcı metodu.
        
        Args:
            model_path (str): Eğitilmiş modelin yolu
            csv_path (str): CSV veri setinin yolu
            host (str): OPC UA sunucusunun IP adresi
            port (int): OPC UA sunucusunun port numarası
            sampling_time (float): Örnekleme zamanı (saniye)
        """
        # Makine kontrol sınıfını oluştur
        self.machine_control = MachineControl(host=host, port=port, sampling_time=sampling_time)
        
        # Veri işleme sınıfını oluştur
        self.data_processor = TemperatureDataProcessor(csv_path, input_window=10, output_window=1, sampling_time=sampling_time)
        self.data_processor.load_data()
        self.data_processor.preprocess_data()
        
        # NARMA-L2 kontrolcüsünü oluştur
        self.controller = NarmaL2Controller(model_path, self.data_processor)
        
        # Kontrol parametreleri
        self.sampling_time = sampling_time
        self.is_connected = False
        
        print("Sıcaklık kontrol sınıfı başlatıldı.")
    
    def connect(self):
        """
        OPC UA sunucusuna bağlanır.
        
        Returns:
            bool: Bağlantı başarılı ise True, değilse False
        """
        if self.machine_control.connect():
            self.is_connected = True
            print("OPC UA sunucusuna başarıyla bağlandı.")
            
            # Güncel sıcaklık ve PWM değerlerini oku
            current_temp = self.machine_control.read_temperature()
            
            if current_temp is None:
                print("Sıcaklık okunamadı. Varsayılan değerler kullanılacak.")
                current_temp = 25.0
                current_pwm = 0.0
            else:
                # PWM değerini doğrudan okuyamıyoruz, bu yüzden 0 olarak varsayalım
                current_pwm = 0.0
            
            # Kontrolcü tamponlarını başlat
            self.controller.initialize_buffers(current_temp, current_pwm)
            
            return True
        else:
            self.is_connected = False
            print("OPC UA sunucusuna bağlanılamadı.")
            return False
    
    def disconnect(self):
        """
        OPC UA sunucusu ile bağlantıyı keser.
        """
        if self.is_connected:
            self.machine_control.disconnect()
            self.is_connected = False
            print("OPC UA sunucusu ile bağlantı kesildi.")
    
    def control_temperature(self, target_temp, duration, log_interval=5):
        """
        Sıcaklığı kontrol eder.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            duration (float): Kontrol süresi (saniye)
            log_interval (int): Kayıt aralığı (saniye)
            
        Returns:
            tuple: Zaman, sıcaklık ve PWM değerlerini içeren diziler
        """
        if not self.is_connected:
            if not self.connect():
                print("OPC UA sunucusuna bağlanılamadı. Kontrol işlemi iptal edildi.")
                return None, None, None
        
        # Sonuçları saklamak için diziler
        time_values = []
        temp_values = []
        pwm_values = []
        
        # Başlangıç zamanı
        start_time = time.time()
        elapsed_time = 0.0
        last_log_time = 0.0
        
        print(f"Sıcaklık kontrolü başlatıldı. Hedef sıcaklık: {target_temp}°C, Süre: {duration} saniye")
        
        try:
            while elapsed_time < duration:
                # Güncel sıcaklığı oku
                current_temp = self.machine_control.read_temperature()
                
                if current_temp is None:
                    print("Sıcaklık okunamadı. Kontrol döngüsü durduruldu.")
                    break
                
                # Kontrol adımını gerçekleştir
                calculated_pwm, predicted_temp = self.controller.control_step(target_temp, current_temp, self.machine_control)
                
                # Sonuçları kaydet
                time_values.append(elapsed_time)
                temp_values.append(current_temp)
                pwm_values.append(calculated_pwm)
                
                # Belirli aralıklarla ilerlemeyi yazdır
                if elapsed_time - last_log_time >= log_interval:
                    print(f"Zaman: {elapsed_time:.1f}s, Sıcaklık: {current_temp:.2f}°C, PWM: {calculated_pwm:.2f}%, "
                          f"Tahmin: {predicted_temp:.2f}°C, Hedef: {target_temp:.2f}°C")
                    last_log_time = elapsed_time
                
                # Örnekleme zamanı kadar bekle
                time.sleep(self.sampling_time)
                
                # Geçen süreyi güncelle
                elapsed_time = time.time() - start_time
            
            print(f"Sıcaklık kontrolü tamamlandı. Toplam süre: {elapsed_time:.1f} saniye")
            
            # Sonuçları NumPy dizilerine dönüştür
            time_values = np.array(time_values)
            temp_values = np.array(temp_values)
            pwm_values = np.array(pwm_values)
            
            return time_values, temp_values, pwm_values
            
        except KeyboardInterrupt:
            print("Kontrol işlemi kullanıcı tarafından durduruldu.")
            
        except Exception as e:
            print(f"Kontrol işlemi sırasında hata oluştu: {e}")
            
        finally:
            # Bağlantıyı kapat
            self.disconnect()
            
            # Sonuçları NumPy dizilerine dönüştür
            time_values = np.array(time_values) if time_values else np.array([])
            temp_values = np.array(temp_values) if temp_values else np.array([])
            pwm_values = np.array(pwm_values) if pwm_values else np.array([])
            
            return time_values, temp_values, pwm_values
    
    def control_to_setpoint(self, target_temp, tolerance=0.5, max_duration=300, stability_duration=30):
        """
        Sıcaklığı belirli bir hedef değere getirir ve o değerde sabit tutar.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            tolerance (float): Tolerans değeri (°C)
            max_duration (float): Maksimum kontrol süresi (saniye)
            stability_duration (float): Kararlılık süresi (saniye)
            
        Returns:
            tuple: Zaman, sıcaklık ve PWM değerlerini içeren diziler
        """
        if not self.is_connected:
            if not self.connect():
                print("OPC UA sunucusuna bağlanılamadı. Kontrol işlemi iptal edildi.")
                return None, None, None
        
        # Sonuçları saklamak için diziler
        time_values = []
        temp_values = []
        pwm_values = []
        
        # Başlangıç zamanı
        start_time = time.time()
        elapsed_time = 0.0
        
        # Kararlılık kontrolü için değişkenler
        stability_start_time = None
        is_stable = False
        
        print(f"Sıcaklık kontrolü başlatıldı. Hedef sıcaklık: {target_temp}°C, Tolerans: ±{tolerance}°C")
        
        try:
            while elapsed_time < max_duration:
                # Güncel sıcaklığı oku
                current_temp = self.machine_control.read_temperature()
                
                if current_temp is None:
                    print("Sıcaklık okunamadı. Kontrol döngüsü durduruldu.")
                    break
                
                # Kontrol adımını gerçekleştir
                calculated_pwm, predicted_temp = self.controller.control_step(target_temp, current_temp, self.machine_control)
                
                # Sonuçları kaydet
                time_values.append(elapsed_time)
                temp_values.append(current_temp)
                pwm_values.append(calculated_pwm)
                
                # Hedef sıcaklığa ulaşıldı mı kontrol et
                if abs(current_temp - target_temp) <= tolerance:
                    # Kararlılık başlangıç zamanını ayarla
                    if stability_start_time is None:
                        stability_start_time = time.time()
                        print(f"Hedef sıcaklığa ulaşıldı: {current_temp:.2f}°C. Kararlılık kontrolü başlatıldı.")
                    
                    # Kararlılık süresini kontrol et
                    stability_elapsed = time.time() - stability_start_time
                    
                    if stability_elapsed >= stability_duration:
                        is_stable = True
                        print(f"Sıcaklık {stability_duration} saniye boyunca kararlı kaldı. Kontrol işlemi başarılı.")
                        break
                else:
                    # Tolerans dışına çıkıldı, kararlılık sayacını sıfırla
                    if stability_start_time is not None:
                        print(f"Sıcaklık tolerans dışına çıktı: {current_temp:.2f}°C. Kararlılık kontrolü sıfırlandı.")
                        stability_start_time = None
                
                # Her 10 saniyede bir ilerlemeyi yazdır
                if int(elapsed_time) % 10 == 0 and int(elapsed_time) > 0:
                    print(f"Zaman: {elapsed_time:.1f}s, Sıcaklık: {current_temp:.2f}°C, PWM: {calculated_pwm:.2f}%, "
                          f"Tahmin: {predicted_temp:.2f}°C, Hedef: {target_temp:.2f}°C")
                
                # Örnekleme zamanı kadar bekle
                time.sleep(self.sampling_time)
                
                # Geçen süreyi güncelle
                elapsed_time = time.time() - start_time
            
            if is_stable:
                print(f"Sıcaklık kontrolü başarıyla tamamlandı. Toplam süre: {elapsed_time:.1f} saniye")
            else:
                print(f"Maksimum süre aşıldı. Kararlı sıcaklık elde edilemedi. Toplam süre: {elapsed_time:.1f} saniye")
            
            # Sonuçları NumPy dizilerine dönüştür
            time_values = np.array(time_values)
            temp_values = np.array(temp_values)
            pwm_values = np.array(pwm_values)
            
            return time_values, temp_values, pwm_values
            
        except KeyboardInterrupt:
            print("Kontrol işlemi kullanıcı tarafından durduruldu.")
            
        except Exception as e:
            print(f"Kontrol işlemi sırasında hata oluştu: {e}")
            
        finally:
            # Bağlantıyı kapat
            self.disconnect()
            
            # Sonuçları NumPy dizilerine dönüştür
            time_values = np.array(time_values) if time_values else np.array([])
            temp_values = np.array(temp_values) if temp_values else np.array([])
            pwm_values = np.array(pwm_values) if pwm_values else np.array([])
            
            return time_values, temp_values, pwm_values
    
    def simulate_control_to_setpoint(self, target_temp, initial_temp, tolerance=0.5, max_duration=300, stability_duration=30):
        """
        Sıcaklık kontrolünü simüle eder.
        
        Args:
            target_temp (float): Hedef sıcaklık değeri
            initial_temp (float): Başlangıç sıcaklık değeri
            tolerance (float): Tolerans değeri (°C)
            max_duration (float): Maksimum simülasyon süresi (saniye)
            stability_duration (float): Kararlılık süresi (saniye)
            
        Returns:
            tuple: Zaman, sıcaklık ve PWM değerlerini içeren diziler
        """
        # Kontrolcü tamponlarını başlat
        self.controller.initialize_buffers(initial_temp, 0.0)
        
        # Sonuçları saklamak için diziler
        time_values = []
        temp_values = []
        pwm_values = []
        
        # Simülasyon başlangıcı
        current_time = 0.0
        current_temp = initial_temp
        
        # Kararlılık kontrolü için değişkenler
        stability_start_time = None
        is_stable = False
        
        print(f"Simülasyon başlatıldı. Hedef sıcaklık: {target_temp}°C, Başlangıç sıcaklığı: {initial_temp}°C")
        
        while current_time < max_duration:
            # Kontrol adımını gerçekleştir
            calculated_pwm, predicted_temp = self.controller.control_step(target_temp, current_temp)
            
            # Sonuçları kaydet
            time_values.append(current_time)
            temp_values.append(current_temp)
            pwm_values.append(calculated_pwm)
            
            # Hedef sıcaklığa ulaşıldı mı kontrol et
            if abs(current_temp - target_temp) <= tolerance:
                # Kararlılık başlangıç zamanını ayarla
                if stability_start_time is None:
                    stability_start_time = current_time
                    print(f"Hedef sıcaklığa ulaşıldı: {current_temp:.2f}°C. Kararlılık kontrolü başlatıldı.")
                
                # Kararlılık süresini kontrol et
                stability_elapsed = current_time - stability_start_time
                
                if stability_elapsed >= stability_duration:
                    is_stable = True
                    print(f"Sıcaklık {stability_duration} saniye boyunca kararlı kaldı. Simülasyon başarılı.")
                    break
            else:
                # Tolerans dışına çıkıldı, kararlılık sayacını sıfırla
                if stability_start_time is not None:
                    print(f"Sıcaklık tolerans dışına çıktı: {current_temp:.2f}°C. Kararlılık kontrolü sıfırlandı.")
                    stability_start_time = None
            
            # Her 10 saniyede bir ilerlemeyi yazdır
            if int(current_time) % 10 == 0 and int(current_time) > 0:
                print(f"Zaman: {current_time:.1f}s, Sıcaklık: {current_temp:.2f}°C, PWM: {calculated_pwm:.2f}%, "
                      f"Tahmin: {predicted_temp:.2f}°C, Hedef: {target_temp:.2f}°C")
            
            # Bir sonraki zaman adımı için sıcaklığı güncelle (basit bir model)
            current_temp = predicted_temp
            
            # Zaman adımını güncelle
            current_time += self.sampling_time
        
        if is_stable:
            print(f"Simülasyon başarıyla tamamlandı. Toplam süre: {current_time:.1f} saniye")
        else:
            print(f"Maksimum süre aşıldı. Kararlı sıcaklık elde edilemedi. Toplam süre: {current_time:.1f} saniye")
        
        # Sonuçları NumPy dizilerine dönüştür
        time_values = np.array(time_values)
        temp_values = np.array(temp_values)
        pwm_values = np.array(pwm_values)
        
        return time_values, temp_values, pwm_values
    
    def plot_results(self, time_values, temp_values, pwm_values, target_temp, save_path='./data/control_results.png'):
        """
        Kontrol sonuçlarını görselleştirir.
        
        Args:
            time_values (numpy.ndarray): Zaman değerleri
            temp_values (numpy.ndarray): Sıcaklık değerleri
            pwm_values (numpy.ndarray): PWM değerleri
            target_temp (float): Hedef sıcaklık değeri
            save_path (str): Grafiğin kaydedileceği yol
        """
        # Hedef sıcaklık dizisi oluştur
        target_temps = np.ones_like(temp_values) * target_temp
        
        # Kontrol performansını görselleştir
        plot_control_performance(
            target_temp=target_temps,
            actual_temp=temp_values,
            pwm_values=pwm_values,
            time_values=time_values,
            save_path=save_path
        )
