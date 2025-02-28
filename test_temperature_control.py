from temperature_control import TemperatureControl
import numpy as np

def main():
    # Sıcaklık kontrol sınıfını oluştur
    temp_control = TemperatureControl(
        model_path='./data/narma_l2_model.pth',
        csv_path='./data/tempdata.csv',
        host="192.168.1.40",
        port=4840,
        sampling_time=0.2
    )
    
    # Hedef sıcaklık
    target_temp = 30.0
    
    # Başlangıç sıcaklığı
    initial_temp = 25.0
    
    # Simülasyon parametreleri
    tolerance = 0.5  # Tolerans (°C)
    max_duration = 120.0  # Maksimum simülasyon süresi (saniye)
    stability_duration = 10.0  # Kararlılık süresi (saniye)
    
    # Simülasyonu çalıştır
    time_values, temp_values, pwm_values = temp_control.simulate_control_to_setpoint(
        target_temp=target_temp,
        initial_temp=initial_temp,
        tolerance=tolerance,
        max_duration=max_duration,
        stability_duration=stability_duration
    )
    
    # Sonuçları görselleştir
    temp_control.plot_results(
        time_values=time_values,
        temp_values=temp_values,
        pwm_values=pwm_values,
        target_temp=target_temp,
        save_path='./data/temperature_control_simulation.png'
    )
    
    print("Test tamamlandı.")

if __name__ == "__main__":
    main()
