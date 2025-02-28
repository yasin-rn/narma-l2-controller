from data_processing.data_processor import TemperatureDataProcessor, create_dataloaders
from models.neural_network import NarmaL2Network, NarmaL2Trainer
from data_processing.visualize import plot_training_history, plot_predictions
import torch

def main():
    # Veri işleme sınıfını oluştur
    data_processor = TemperatureDataProcessor('./data/tempdata.csv', input_window=10, output_window=1)
    
    # Verileri yükle ve ön işleme tabi tut
    data_processor.load_data()
    data_processor.preprocess_data()
    
    # Eğitim ve test dizileri oluştur
    X_train, y_train, X_test, y_test = data_processor.create_sequences(train_size=0.8)
    
    # DataLoader nesneleri oluştur
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    # Model parametreleri
    input_size = X_train.shape[1]  # Giriş boyutu
    hidden_size = 64  # Gizli katman boyutu
    output_size = y_train.shape[1]  # Çıkış boyutu
    
    # Modeli oluştur
    model = NarmaL2Network(input_size, hidden_size, output_size, num_layers=2, dropout=0.2)
    
    # Trainer oluştur
    trainer = NarmaL2Trainer(model)
    
    # Modeli eğit
    train_losses, val_losses = trainer.train(
        train_loader, 
        test_loader,  # Test veri setini doğrulama seti olarak kullan
        num_epochs=50,
        learning_rate=0.001,
        weight_decay=1e-5,
        patience=10
    )
    
    # Eğitim geçmişini görselleştir
    plot_training_history(train_losses, val_losses)
    
    # Modeli değerlendir
    y_true, y_pred, test_loss = trainer.evaluate(test_loader)
    
    # Normalize edilmiş değerleri orijinal ölçeğe dönüştür
    y_true_orig = data_processor.inverse_transform_temp(y_true)
    y_pred_orig = data_processor.inverse_transform_temp(y_pred)
    
    # Tahminleri görselleştir
    plot_predictions(y_true_orig, y_pred_orig)
    
    # Modeli kaydet
    trainer.save_model()
    
    print("Test tamamlandı.")

if __name__ == "__main__":
    main()
