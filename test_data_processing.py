from data_processing.data_processor import TemperatureDataProcessor
import matplotlib.pyplot as plt

def main():
    # Veri işleme sınıfını oluştur
    data_processor = TemperatureDataProcessor('tempdata.csv', input_window=10, output_window=1)
    
    # Verileri yükle
    data = data_processor.load_data()
    if data is None:
        print("Veri yüklenemedi.")
        return
    
    # Verileri ön işleme tabi tut
    processed_data, _, _ = data_processor.preprocess_data()
    if processed_data is None:
        print("Veri ön işleme hatası.")
        return
    
    # Veriyi görselleştir
    data_processor.plot_data()
    
    # Eğitim ve test dizileri oluştur
    X_train, y_train, X_test, y_test = data_processor.create_sequences(train_size=0.8)
    if X_train is None:
        print("Diziler oluşturulamadı.")
        return
    
    print("Veri işleme tamamlandı.")
    print(f"Eğitim seti boyutu: {X_train.shape}, {y_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}, {y_test.shape}")

if __name__ == "__main__":
    main()
