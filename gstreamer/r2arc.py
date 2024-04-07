# r2arc.py
# Main program for the R2ARC project

import detect
import ble
import enum

class Controls:
    FORWARD = 'W'
    LEFT = 'A'
    BACKWARD = 'S'
    RIGHT = 'D'
    STOP = 'Q'
    FOLLOW = 'F'
    REMOTE = 'R'
    MOVEMENTS = {FORWARD, 
                LEFT, 
                BACKWARD, 
                RIGHT, 
                STOP}
    STATES = {STOP,
            FOLLOW,
            REMOTE}
    ALL = STATES.union(MOVEMENTS)

class State(enum.Enum):
    IDLE = 0
    FOLLOW = 1
    REMOTE = 2

def get_command():
    """Get a command from the user."""
    command = ''
    while command not in Controls.ALL:
        command = input("Enter a command: ").upper()
    print(f'Command was a valid command: {command}')
    return command

if __name__ == '__main__':
    SERVICE_UUID = '12345678-1234-1234-1234-123456789012'
    CHARACTERISTIC_UUID = '87654321-4321-4321-4321-210987654321'
    r2ble = ble.R2ARC_Service(SERVICE_UUID, CHARACTERISTIC_UUID)
    r2ble.setup()
    state = State.IDLE
    command = ''

    try:
        while True:
            command = r2ble.update_user_input()
            if command == Controls.STOP:
                state = State.IDLE
            elif command == Controls.FOLLOW:
                state = State.FOLLOW
            elif command == Controls.REMOTE or command in Controls.MOVEMENTS:
                state = State.REMOTE
            else:
                print("Invalid command")
            print(f"State: {state}, Command: {command}")

            # if state == State.FOLLOW:
                # detect.main()
    except KeyboardInterrupt:
        print("Program manually terminated")
        r2ble.stop()