import spidev
import time

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

    # Below methods implement specific movements by sending single-byte commands
    def forward(self) -> list[int]:
        """"Sends the character 'W' to the motor controller via SPI to move the R2-ARC droid forward.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """ 
        return self.spi.xfer([ord('W')])

    def left(self) -> list[int]:
        """"Sends the character 'A' to the motor controller via SPI to move the R2-ARC droid left.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord('A')])

    def backwards(self) -> list[int]:
        """"Sends the character 'S' to the motor controller via SPI to move the R2-ARC droid backwards.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord('S')])

    def right(self) -> list[int]:
        """"Sends the character 'D' to the motor controller via SPI to move the R2-ARC droid right.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord('D')])

    def pivot_left(self) -> list[int]:
        """"Sends the character 'O' to the motor controller via SPI to pivot the R2-ARC droid left in place.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord('O')])

    def pivot_right(self) -> list[int]:
        """"Sends the character 'P' to the motor controller via SPI to pivot the R2-ARC droid right in place.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord('P')])

    def stop(self) -> list[int]:
        """"Sends the character 'Q' to the motor controller via SPI to stop the R2-ARC droid.
        args:
            None
        returns:
            list[int]: List containing the received byte from the motor controller.
        """
        return self.spi.xfer([ord('Q')])

def test_commands() -> None:
    """Test sending a series of commands with 2 second delays to the motor controller.
    args:
        None
    returns:
        None
    """
    move = Movements()
    commands = {"W": move.forward, 
                "A": move.left, 
                "S": move.backwards, 
                "D": move.right, 
                "O": move.pivot_left, 
                "P": move.pivot_right, 
                "Q": move.stop}
    for key, command in commands.items():
        print(f"Attempting to send command: {key}")
        received = command()
        print(f"Sent: {key}, Received Data: 0x{received[0]:02x}, {chr(received[0])}")
        time.sleep(2)

if __name__ == "__main__":
    test_commands()
