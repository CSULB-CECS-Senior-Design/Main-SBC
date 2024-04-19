'''
r2arc.py
Main program for the R2ARC project
'''

import vision, motors, ble
import enum, threading
from gstreamer import cameras

class Controls:
    FORWARD = 'W'
    LEFT = 'A'
    BACKWARD = 'S'
    RIGHT = 'D'
    PIVOT_LEFT = 'O'
    PIVOT_RIGHT = 'P'
    STOP = 'Q'
    FOLLOW = 'F'
    REMOTE = 'R'
    MOVEMENTS: set = {
        FORWARD, 
        LEFT, 
        BACKWARD, 
        RIGHT, 
        PIVOT_LEFT, 
        PIVOT_RIGHT, 
        STOP
    }
    STATES: set = {
        STOP,
        FOLLOW, 
        REMOTE
    }
    ALL: set = STATES.union(MOVEMENTS)
    prints: dict = {
        FORWARD: "Moving forward", 
        LEFT: "Moving left", 
        BACKWARD: "Moving backward", 
        RIGHT: "Moving right", 
        PIVOT_LEFT: "Pivoting left", 
        PIVOT_RIGHT: "Pivoting right", 
        STOP: "Stopping", 
        FOLLOW: "Following", 
        REMOTE: "Remote control"
    }

    def print_valid_command(command: str) -> str:
        if command in Controls.ALL:
            return Controls.prints[command]
        return "Invalid command"

class State(enum.Enum):
    IDLE = 0
    FOLLOW = 1
    REMOTE = 2

if __name__ == '__main__':
    # Setup motor controller communication
    r2motor = motors.Movements()
    # Setup Machine Vision
    r2vision = vision.DroidVision(resolution=cameras.get_razer_kiyo_resolution(), motor=r2motor)
    r2vision_thread = threading.Thread(target=r2vision.start)
    r2vision_thread.start()
    # Setup Bluetooth Low Energy connection
    SERVICE_UUID = '12345678-1234-1234-1234-123456789012'
    CHARACTERISTIC_UUID = '87654321-4321-4321-4321-210987654321'
    r2ble = ble.R2ARCService(SERVICE_UUID, CHARACTERISTIC_UUID)
    r2ble.setup()
    # Keep track of states and last command
    state = State.IDLE
    command = ''

    try:
        while True:
            command = r2ble.update_user_input()

            # Update state
            if command == Controls.STOP:
                state = State.IDLE
                r2vision.set_follow(False)
                r2motor.stop()
            elif command == Controls.FOLLOW:
                state = State.FOLLOW if state != State.FOLLOW else State.IDLE
                r2vision.toggle_follow()
            elif command == Controls.REMOTE or command in Controls.MOVEMENTS:
                state = State.REMOTE
                r2vision.set_follow(False)
                r2motor.send_command(command)
            else:
                print("Invalid command")

            print(f"Command function: {Controls.print_valid_command(command)}, State: {state} \n")

    except KeyboardInterrupt:
        print("Quitting program")
        r2motor.stop()
        r2ble.stop()
        r2vision.stop(process="r2arc.py")
        r2vision_thread.join()
        