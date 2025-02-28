from machine_control.machine_control import MachineControl
import time


def main():
    # MachineControl sınıfını oluştur
    machine = MachineControl(host="192.168.1.40", port=4840, sampling_time=0.2)

    # OPC UA sunucusuna bağlan
    if machine.connect():
        print("OPC UA sunucusuna başarıyla bağlandı.")

        try:
            # Sıcaklık değerini oku
            temperature = machine.read_temperature()
            print(f"Anlık sıcaklık: {temperature}")

            # PWM değerini ayarla (örnek olarak %50)
            if machine.set_pwm(1.0):
                print("PWM değeri başarıyla ayarlandı.")

            # Bir süre bekle
            time.sleep(2)

            # Tekrar sıcaklık değerini oku
            temperature = machine.read_temperature()
            print(f"Yeni sıcaklık: {temperature}")

        except Exception as e:
            print(f"Hata oluştu: {e}")

        finally:
            # Bağlantıyı kapat
            machine.disconnect()
    else:
        print("OPC UA sunucusuna bağlanılamadı.")


if __name__ == "__main__":
    main()
