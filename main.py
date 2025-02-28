from temperature_control import TemperatureControl
import time
import argparse
import os


def main():
    # Komut satırı argümanlarını tanımla
    parser = argparse.ArgumentParser(description='NARMA-L2 Sıcaklık Kontrolcüsü')
    parser.add_argument('--target', type=float, default=30.0, help='Hedef sıcaklık (°C)')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Sıcaklık toleransı (°C)')
    parser.add_argument('--max-duration', type=float, default=300.0, help='Maksimum kontrol süresi (saniye)')
    parser.add_argument('--stability-duration', type=float, default=30.0, help='Kararlılık süresi (saniye)')
    parser.add_argument('--host', type=str, default='192.168.1.40', help='OPC UA sunucu adresi')
    parser.add_argument('--port', type=int, default=4840, help='OPC UA sunucu portu')
    parser.add_argument('--model-path', type=str, default='./data/narma_l2_model.pth', help='Model dosyası yolu')
    parser.add_argument('--csv-path', type=str, default='./data/tempdata.csv', help='CSV veri dosyası yolu')
    parser.add_argument('--simulate', action='store_true', help='Simülasyon modunda çalıştır')
    
    args = parser.parse_args()
    
    # Veri ve model dizinlerinin varlığını kontrol et
    os.makedirs('./data', exist_ok=True)
    
    # Sıcaklık kontrol sınıfını oluştur
    temp_control = TemperatureControl(
        model_path=args.model_path,
        csv_path=args.csv_path,
        host=args.host,
        port=args.port,
        sampling_time=0.2
    )
    
    print(f"Hedef sıcaklık: {args.target}°C")
    print(f"Tolerans: {args.tolerance}°C")
    print(f"Maksimum süre: {args.max_duration} saniye")
    print(f"Kararlılık süresi: {args.stability_duration} saniye")
    
    try:
        if args.simulate:
            print("Simülasyon modunda çalışılıyor...")
            # Simülasyon modunda çalıştır
            time_values, temp_values, pwm_values = temp_control.simulate_control_to_setpoint(
                target_temp=args.target,
                initial_temp=25.0,  # Başlangıç sıcaklığı
                tolerance=args.tolerance,
                max_duration=args.max_duration,
                stability_duration=args.stability_duration
            )
        else:
            print("Gerçek sistem modunda çalışılıyor...")
            # OPC UA sunucusuna bağlan
            if not temp_control.connect():
                print("OPC UA sunucusuna bağlanılamadı.")
                return
            
            print("OPC UA sunucusuna başarıyla bağlandı.")
            
            # Gerçek sistemde sıcaklık kontrolü yap
            time_values, temp_values, pwm_values = temp_control.control_to_setpoint(
                target_temp=args.target,
                tolerance=args.tolerance,
                max_duration=args.max_duration,
                stability_duration=args.stability_duration
            )
        
        # Sonuçları görselleştir
        temp_control.plot_results(
            time_values=time_values,
            temp_values=temp_values,
            pwm_values=pwm_values,
            target_temp=args.target,
            save_path='./data/temperature_control_results.png'
        )
        
        print(f"Kontrol tamamlandı. Sonuçlar './data/temperature_control_results.png' dosyasına kaydedildi.")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
    
    finally:
        # Bağlantıyı kapat
        if not args.simulate:
            temp_control.disconnect()


if __name__ == "__main__":
    main()
