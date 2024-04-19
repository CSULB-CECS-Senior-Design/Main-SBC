'''
motors.py
Motor controller communication via SPI for the R2-ARC project
'''

import spidev

class Movements:
    def __init__(self, spi_channel: int = 0, speed: int = 5000000) -> None:
        """Initializes the Movements class with the specified SPI channel and speed to communicate with the motor controller.
        args:
            spi_channel (int): SPI channel (bus) to use. Default is 0.
            speed (int): SPI communication speed in Hz. Default is 5 MHz (5,000,000 Hz).
        returns:
            None
        """
        self.spi_channel = spi_channel
        self.speed = speed
        self.spi = spidev.SpiDev()
        self.spi.open(0, spi_channel)  # Open SPI port 0, chip select (CS) is set by spi_channel
        self.spi.max_speed_hz = speed
        # Store ASCII values for faster communication
        self._W_ASCII = ord('W')
        self._A_ASCII = ord('A')
        self._S_ASCII = ord('S')
        self._D_ASCII = ord('D')
        self._O_ASCII = ord('O')
        self._P_ASCII = ord('P')
        self._Q_ASCII = ord('Q')

    # Below methods implement specific movements by sending single-byte commands
    def forward(self) -> list[int]:
        """"Sends the character 'W' to the motor controller via SPI to move the R2-ARC droid forward.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """ 
        return self.spi.xfer([self._W_ASCII])

    def left(self) -> list[int]:
        """"Sends the character 'A' to the motor controller via SPI to move the R2-ARC droid left.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([self._A_ASCII])

    def backwards(self) -> list[int]:
        """"Sends the character 'S' to the motor controller via SPI to move the R2-ARC droid backwards.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([self._S_ASCII])

    def right(self) -> list[int]:
        """"Sends the character 'D' to the motor controller via SPI to move the R2-ARC droid right.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([self._D_ASCII])

    def pivot_left(self) -> list[int]:
        """"Sends the character 'O' to the motor controller via SPI to pivot the R2-ARC droid left in place.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([self._O_ASCII])

    def pivot_right(self) -> list[int]:
        """"Sends the character 'P' to the motor controller via SPI to pivot the R2-ARC droid right in place.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([self._P_ASCII])

    def stop(self) -> list[int]:
        """"Sends the character 'Q' to the motor controller via SPI to stop the R2-ARC droid.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([self._Q_ASCII])
    
    def send_command(self, command: str) -> list[int]:
        """Sends the specified command to the motor controller.
        args:
            command (str): The command to send to the motor controller.
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord(command)])

if __name__ == "__main__":
    import time
    # Test the Movements class
    move = Movements()
    commands = {
        'W': move.forward, 
        'A': move.left, 
        'S': move.backwards, 
        'D': move.right, 
        'O': move.pivot_left, 
        'P': move.pivot_right, 
        'Q': move.stop
    }
    for key, command in commands.items():
        print(f"Attempting to send command: {key}")
        received = command()
        print(f"Sent: {key}, Received Data: 0x{received[0]:02x}, {chr(received[0])}")
        time.sleep(4)

    for dec in range(256):
        print(f"Attempting to send command: {chr(dec)}")
        received = move.send_command(chr(dec))
        print(f"Sent: {chr(dec)}, Received Data: 0x{received[0]:02x}, {chr(received[0])}")
        time.sleep(4)