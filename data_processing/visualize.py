import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """
    Eğitim geçmişini görselleştirir.
    
    Args:
        train_losses (list): Eğitim kayıpları
        val_losses (list): Doğrulama kayıpları
        save_path (str): Grafiğin kaydedileceği yol
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Eğitim Kaybı')
    plt.plot(val_losses, label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Eğitim ve Doğrulama Kayıpları')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Eğitim geçmişi grafiği kaydedildi: {save_path}")

def plot_predictions(y_true, y_pred, time_values=None, save_path='predictions.png'):
    """
    Gerçek ve tahmin edilen değerleri görselleştirir.
    
    Args:
        y_true (numpy.ndarray): Gerçek değerler
        y_pred (numpy.ndarray): Tahmin edilen değerler
        time_values (numpy.ndarray, optional): Zaman değerleri
        save_path (str): Grafiğin kaydedileceği yol
    """
    plt.figure(figsize=(12, 6))
    
    if time_values is None:
        time_values = np.arange(len(y_true))
    
    plt.plot(time_values, y_true, 'b-', label='Gerçek Sıcaklık')
    plt.plot(time_values, y_pred, 'r--', label='Tahmin Edilen Sıcaklık')
    plt.xlabel('Zaman')
    plt.ylabel('Sıcaklık (°C)')
    plt.title('Gerçek ve Tahmin Edilen Sıcaklık Değerleri')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Tahmin grafiği kaydedildi: {save_path}")

def plot_control_performance(target_temp, actual_temp, pwm_values, time_values=None, save_path='control_performance.png'):
    """
    Kontrol performansını görselleştirir.
    
    Args:
        target_temp (numpy.ndarray): Hedef sıcaklık değerleri
        actual_temp (numpy.ndarray): Gerçek sıcaklık değerleri
        pwm_values (numpy.ndarray): PWM değerleri
        time_values (numpy.ndarray, optional): Zaman değerleri
        save_path (str): Grafiğin kaydedileceği yol
    """
    if time_values is None:
        time_values = np.arange(len(actual_temp))
    
    plt.figure(figsize=(12, 8))
    
    # Sıcaklık grafiği
    plt.subplot(2, 1, 1)
    plt.plot(time_values, target_temp, 'g-', label='Hedef Sıcaklık')
    plt.plot(time_values, actual_temp, 'b-', label='Gerçek Sıcaklık')
    plt.ylabel('Sıcaklık (°C)')
    plt.title('Kontrol Performansı')
    plt.grid(True)
    plt.legend()
    
    # PWM grafiği
    plt.subplot(2, 1, 2)
    plt.plot(time_values, pwm_values, 'r-', label='PWM')
    plt.xlabel('Zaman')
    plt.ylabel('PWM (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Kontrol performansı grafiği kaydedildi: {save_path}")
