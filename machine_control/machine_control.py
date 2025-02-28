from opcua import Client
import time

class MachineControl:
    """
    OPC UA bağlantısını sağlayan sınıf.
    Bu sınıf, sıcaklık okuma ve PWM değerini ayarlama işlemlerini gerçekleştirir.
    """
    
    def __init__(self, host="192.168.1.40", port=4840, sampling_time=0.2):
        """
        MachineControl sınıfının yapıcı metodu.
        
        Args:
            host (str): OPC UA sunucusunun IP adresi
            port (int): OPC UA sunucusunun port numarası
            sampling_time (float): Örnekleme zamanı (saniye)
        """
        self.host = host
        self.port = port
        self.sampling_time = sampling_time
        self.client = None
        self.temp_node = None
        self.pwm_node = None
        self.connected = False
        
    def connect(self):
        """
        OPC UA sunucusuna bağlanır ve gerekli node'ları alır.
        
        Returns:
            bool: Bağlantı başarılı ise True, değilse False
        """
        try:
            # OPC UA sunucusuna bağlan
            server_url = f"opc.tcp://{self.host}:{self.port}"
            self.client = Client(server_url)
            self.client.connect()
            
            # Sıcaklık ve PWM node'larını al
            self.temp_node = self.client.get_node("ns=4;s=|var|AC500 PM56xx-2ETH.Application.OPC_UA.Active_Temp")
            self.pwm_node = self.client.get_node("ns=4;s=|var|AC500 PM56xx-2ETH.Application.OPC_UA.Pwm_Value")
            
            self.connected = True
            return True
        except Exception as e:
            print(f"Bağlantı hatası: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        OPC UA sunucusu ile bağlantıyı keser.
        """
        if self.client and self.connected:
            self.client.disconnect()
            self.connected = False
            print("OPC UA sunucusu ile bağlantı kesildi.")
    
    def read_temperature(self):
        """
        Sistemin anlık sıcaklığını okur.
        
        Returns:
            float: Okunan sıcaklık değeri, bağlantı yoksa None
        """
        if not self.connected:
            print("OPC UA sunucusuna bağlı değil. Önce connect() metodunu çağırın.")
            return None
        
        try:
            temperature = self.temp_node.get_value()
            return float(temperature)
        except Exception as e:
            print(f"Sıcaklık okuma hatası: {e}")
            return None
    
    def set_pwm(self, pwm_value):
        """
        PWM değerini ayarlar.
        
        Args:
            pwm_value (float): Ayarlanacak PWM değeri
            
        Returns:
            bool: İşlem başarılı ise True, değilse False
        """
        if not self.connected:
            print("OPC UA sunucusuna bağlı değil. Önce connect() metodunu çağırın.")
            return False
        
        try:
            # PWM değerini 'single' (float32) veri türüne dönüştür
            from opcua import ua
            dv = ua.DataValue(ua.Variant(float(pwm_value), ua.VariantType.Float))
            self.pwm_node.set_value(dv)
            return True
        except Exception as e:
            print(f"PWM ayarlama hatası: {e}")
            return False
    
    def __del__(self):
        """
        Nesne silindiğinde bağlantıyı otomatik olarak kapat.
        """
        self.disconnect()
