# r2arc.py
# Main program for the R2ARC project

import detect
import ble
import enum
import threading
import cameras
import motors

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
    MOVEMENTS = {FORWARD, 
                LEFT, 
                BACKWARD, 
                RIGHT, 
                PIVOT_LEFT, 
                PIVOT_RIGHT, 
                STOP}
    STATES = {STOP,
            FOLLOW,
            REMOTE}
    ALL = STATES.union(MOVEMENTS)

class State(enum.Enum):
    IDLE = 0
    FOLLOW = 1
    REMOTE = 2

if __name__ == '__main__':
    # Setup Machine Vision
    vision = detect.DroidVision(resolution=cameras.get_razer_kiyo_resolution())
    vision_thread = threading.Thread(target=vision.start)
    vision_thread.start()
    # Setup Bluetooth Low Energy connection
    SERVICE_UUID = '12345678-1234-1234-1234-123456789012'
    CHARACTERISTIC_UUID = '87654321-4321-4321-4321-210987654321'
    r2ble = ble.R2ARC_Service(SERVICE_UUID, CHARACTERISTIC_UUID)
    r2ble.setup()
    # Keep track of states and last command
    state = State.IDLE
    command = ''
    moves = motors.Movements()

    def move_motor(command: str) -> None:
        if command == Controls.FORWARD:
            moves.forward()
        elif command == Controls.LEFT:
            moves.left()
        elif command == Controls.BACKWARD:
            moves.backwards()
        elif command == Controls.RIGHT:
            moves.right()
        elif command == Controls.STOP:
            moves.stop()
        elif command == Controls.PIVOT_LEFT:
            moves.pivot_left()
        elif command == Controls.PIVOT_RIGHT:
            moves.pivot_right()

    try:
        while True:
            command = r2ble.update_user_input()
            # Update state
            if command == Controls.STOP:
                state = State.IDLE
                vision.set_follow(False)
                moves.stop()
            elif command == Controls.FOLLOW:
                state = State.FOLLOW if state != State.FOLLOW else State.IDLE
                vision.toggle_follow()
            elif command == Controls.REMOTE or command in Controls.MOVEMENTS:
                state = State.REMOTE
                vision.set_follow(False)
                move_motor(command)
            else:
                print("Invalid command")

            print(f"State: {state}, Command: {command}")

    except KeyboardInterrupt:
        # vision.stop()
        vision_thread.join()
        print("Program manually terminated")
        r2ble.stop()