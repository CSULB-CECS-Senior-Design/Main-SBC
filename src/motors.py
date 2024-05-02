'''
motors.py
Motor controller communication via SPI for the R2-ARC project
'''

import spidev

class Movements:
    def __init__(self, spi: spidev.SpiDev) -> None:
        """Initializes the Movements class with the specified SPI object.
        args:
            spi (spidev.SpiDev): The SPI object to use for communication with the motor controller.
        """
        self.spi = spi
        # Store ASCII values for faster communication
        self._W_ASCII = ord('W')
        self._A_ASCII = ord('A')
        self._S_ASCII = ord('S')
        self._D_ASCII = ord('D')
        self._O_ASCII = ord('O')
        self._P_ASCII = ord('P')
        self._Q_ASCII = ord('Q')
        self._EMPTY_BYTE = 0x00

    # Below methods implement specific movements by sending single-byte commands
    def forward(self) -> list[int]:
        """"Sends the character 'W' to the motor controller via SPI to move the R2-ARC droid forward.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """ 
        return self.spi.xfer([self._EMPTY_BYTE, self._W_ASCII])

    def left(self) -> list[int]:
        """"Sends the character 'A' to the motor controller via SPI to move the R2-ARC droid left.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, self._A_ASCII])

    def backward(self) -> list[int]:
        """"Sends the character 'S' to the motor controller via SPI to move the R2-ARC droid backwards.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, self._S_ASCII])

    def right(self) -> list[int]:
        """"Sends the character 'D' to the motor controller via SPI to move the R2-ARC droid right.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, self._D_ASCII])

    def pivot_left(self) -> list[int]:
        """"Sends the character 'O' to the motor controller via SPI to pivot the R2-ARC droid left in place.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, self._O_ASCII])

    def pivot_right(self) -> list[int]:
        """"Sends the character 'P' to the motor controller via SPI to pivot the R2-ARC droid right in place.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, self._P_ASCII])

    def stop(self) -> list[int]:
        """"Sends the character 'Q' to the motor controller via SPI to stop the R2-ARC droid.
        args:
            None
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, self._Q_ASCII])
    
    def send_command(self, command: str) -> list[int]:
        """Sends the specified command to the motor controller.
        args:
            command (str): The command to send to the motor controller.
        returns:
            list[int]: List containing the the relayed InfraRed sensor encoded 16-bit data.
        """
        return self.spi.xfer([self._EMPTY_BYTE, ord(command)])

if __name__ == "__main__":
    import time
    # Test the Movements class
    spi = spidev.SpiDev()
    spi.open(bus=0, device=0)
    spi.max_speed_hz = 5_000_000
    move = Movements(spi)
    commands = {
        'W': move.forward, 
        'A': move.left, 
        'S': move.backward, 
        'D': move.right, 
        'O': move.pivot_left, 
        'P': move.pivot_right, 
        'Q': move.stop
    }
    for key, command in commands.items():
        print(f"Attempting to send command: {key}")
        received = command()
        print(f"Sent: {key}, Received Data: {received}")
        time.sleep(1)

    for num in range(256):
        print(f"Attempting to send command: {chr(num)}")
        received = move.send_command(chr(num))
        print(f"Sent: {chr(num)}, Received Data: {received}")
        time.sleep(1)