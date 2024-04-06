import os
import threading
import time
import pybleno
import array

class SimpleCharacteristic(pybleno.Characteristic):
    """A custom characteristic for handling write requests."""
    def __init__(self, uuid):
        super().__init__({
            'uuid': uuid,
            'properties': ['write'],
            'value': None
        })
        self._value = array.array('B', [0]*0)  # Initialize with an empty buffer

    def onWriteRequest(self, data, offset, withoutResponse, callback):
        self._value = data  # Update the value with incoming data
        print(f'Received message: {self._value.decode("utf-8")}')
        callback(pybleno.Characteristic.RESULT_SUCCESS)

class R2ARC_Service:
    """A class to handle the BLE service and characteristic."""
    def __init__(self, service_uuid, characteristic_uuid):
        self._service_uuid = service_uuid
        self._characteristic_uuid = characteristic_uuid
        self.reset_bluetooth()
        self._bleno = pybleno.Bleno()

    def get_characteristic_uuid(self):
        return self._characteristic_uuid
    
    def get_service_uuid(self):
        return self._service_uuid

    def reset_bluetooth(self, delay: float = 0.2) -> None:
        """Resets the Bluetooth service on the system."""
        print("Resetting Bluetooth")
        os.system('sudo systemctl stop bluetooth')
        time.sleep(delay)  # Short pause
        os.system('sudo systemctl start bluetooth')
        time.sleep(delay)  # Wait for Bluetooth service to restart

    def reset_ble_services(self):
        """
        Resets the Pybleno instance and restarts the Bluetooth service.
        Will throw an error; used for testing.
        """
        print("Stopping Pybleno")
        self._bleno.stopAdvertising()
        self._bleno.disconnect()
        self._bleno = None  # Clear the instance

        self.reset_bluetooth()  # Restart the Bluetooth service

        print("Restarting Pybleno")
        self._bleno = pybleno.Bleno()  # Reinitialize Pybleno
        self._setup_bleno()  # Setup Bleno with services and characteristics

    def _setup_bleno(self):
        """Sets up Bleno with event listeners and starts advertising."""
        self._bleno.on('stateChange', self._onStateChange)
        self._bleno.on('advertisingStart', self._onAdvertisingStart)
        self._bleno.start()

    def _onStateChange(self, state):
        """Handles state changes in Bleno."""
        print(f'BLE state changed to {state}')
        if state == 'poweredOn':
            self._bleno.startAdvertising('RaspberryPi', [self._service_uuid])
        else:
            self._bleno.stopAdvertising()

    def _onAdvertisingStart(self, error):
        """Callback for when BLE advertising starts."""
        print('Advertising start: ' + ('error ' + str(error) if error else 'success'))
        if not error:
            self._bleno.setServices([
                pybleno.BlenoPrimaryService({
                    'uuid': self._service_uuid,
                    'characteristics': [SimpleCharacteristic(self._characteristic_uuid)]
                })
            ])

    def start(self):
        """Starts the BLE service."""
        self._setup_bleno()

        try:
            while True:
                pass
        except KeyboardInterrupt:
            # Graceful shutdown on keyboard interrupt
            print("Program manually terminated")
            self._bleno.stopAdvertising()
