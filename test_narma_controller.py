from data_processing.data_processor import TemperatureDataProcessor
from models.neural_network import NarmaL2Network, NarmaL2Trainer
from controllers.narma_l2_controller import NarmaL2Controller
from data_processing.visualize import plot_control_performance
import numpy as np

def main():
    # Veri işleme sınıfını oluştur
    data_processor = TemperatureDataProcessor('tempdata.csv', input_window=10, output_window=1, sampling_time=0.2)
    
    # Verileri yükle ve ön işleme tabi tut
    data_processor.load_data()
    data_processor.preprocess_data()
    
    # NARMA-L2 kontrolcüsünü oluştur
    controller = NarmaL2Controller('narma_l2_model.pth', data_processor)
    
    # Kontrol sistemini simüle et
    target_temp = 30.0  # Hedef sıcaklık
    initial_temp = 25.0  # Başlangıç sıcaklığı
    duration = 60.0  # Simülasyon süresi (saniye)
    
    # Simülasyonu çalıştır
    time_values, temp_values, pwm_values = controller.simulate_control(
        target_temp=target_temp,
        initial_temp=initial_temp,
        duration=duration
    )
    
    # Kontrol performansını görselleştir
    target_temps = np.ones_like(temp_values) * target_temp
    plot_control_performance(
        target_temp=target_temps,
        actual_temp=temp_values,
        pwm_values=pwm_values,
        time_values=time_values,
        save_path='control_simulation.png'
    )
    
    print("Test tamamlandı.")

if __name__ == "__main__":
    main()
