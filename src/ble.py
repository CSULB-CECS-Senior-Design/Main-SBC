'''
ble.py
Bluetooth Low Energy (BLE) service for the R2-ARC project. Uses iOS app to send commands to the main SBC.
'''

import os, time, pybleno, array, subprocess

class RecieveCharactersCharacteristic(pybleno.Characteristic):
    """A custom characteristic for handling write requests through iOS BLE."""
    def __init__(self, uuid):
        super().__init__({
            'uuid': uuid,
            'properties': ['write'],
            'value': None
        })
        self._value = array.array('B', [0]*0)  # Initialize with an empty buffer


    def onWriteRequest(self, data, offset, withoutResponse, callback) -> None:
        self._value = data  # Update the value with incoming data
        print(f'Received message: {self._value}, decoded: {self._value.decode("utf-8")}')
        callback(pybleno.Characteristic.RESULT_SUCCESS)

    def getValue(self) -> str:
        while not self._value:
            pass
        return self._value.decode('utf-8')
    
    def resetValue(self) -> None:
        self._value = array.array('B', [0]*0)

class R2ARCService:
    """A class to handle the BLE service and characteristic."""
    def __init__(self, service_uuid, characteristic_uuid):
        self._service_uuid = service_uuid
        self._characteristic_uuid = characteristic_uuid
        self.characteristic = RecieveCharactersCharacteristic(characteristic_uuid)
        self.reset_bluetooth()
        self._bleno = pybleno.Bleno()
        self.ready = False

    def get_characteristic_uuid(self):
        return self._characteristic_uuid
    
    def get_service_uuid(self):
        return self._service_uuid
    
    def stop(self):
        print("Stopping Pybleno")
        self._bleno.stopAdvertising()
        self._bleno.disconnect()
        self._bleno = None
        self.ready = False

    def reset_bluetooth(self, delay: float = 0.2) -> None:
        """Resets the Bluetooth service on the system."""
        print("Resetting Bluetooth")

        self.ready = False
        subprocess.run(['sudo', 'hciconfig', 'hci0', 'reset'], check=True)
        self.ready = True

        # os.system('sudo systemctl stop bluetooth')
        # self.ready = False
        # time.sleep(delay)  # Short pause
        # os.system('sudo systemctl start bluetooth')
        # time.sleep(delay)  # Wait for Bluetooth service to restart
        # self.ready = True

    def reset_ble_services(self):
        """
        Resets the Pybleno instance and restarts the Bluetooth service.
        Will throw an error; used for testing.
        """
        self.stop()  # Stop the Pybleno instance

        self.reset_bluetooth()  # Restart the Bluetooth service

        print("Restarting Pybleno")
        self._bleno = pybleno.Bleno()  # Reinitialize Pybleno
        self.setup()  # Setup Bleno with services and characteristics

    def _on_state_change(self, state):
        """Handles state changes in Bleno."""
        print(f'BLE state changed to {state}')
        if state == 'poweredOn':
            self._bleno.startAdvertising('RaspberryPi', [self._service_uuid])
        else:
            self._bleno.stopAdvertising()

    def _on_advertising_start(self, error):
        """Callback for when BLE advertising starts."""
        print('Advertising start: ' + ('error ' + str(error) if error else 'success'))
        if not error:
            self._bleno.setServices([
                pybleno.BlenoPrimaryService({
                    'uuid': self._service_uuid,
                    'characteristics': [self.characteristic]
                })
            ])
            self.ready = True

    def setup(self):
        """Sets up Bleno with event listeners and starts advertising."""
        self._bleno.on('stateChange', self._on_state_change)
        self._bleno.on('advertisingStart', self._on_advertising_start)
        self._bleno.start()
        while not self.ready:
            pass

    def update_user_input(self) -> str:
        data = self.characteristic.getValue()
        self.characteristic.resetValue()
        return data
    
if __name__ == '__main__':
    # Test out the BLE service
    SERVICE_UUID = '12345678-1234-1234-1234-123456789012'
    CHARACTERISTIC_UUID = '87654321-4321-4321-4321-210987654321'
    r2ble = R2ARCService(SERVICE_UUID, CHARACTERISTIC_UUID)
    r2ble.setup()
    
    try:
        while True:
            command = r2ble.update_user_input()
            print(f"Command: {command}")

    except KeyboardInterrupt:
        print("Quitting program")
        r2ble.stop()
    