import cv2

def get_resolution(camera_id: int = 0) -> tuple:
    '''
    Get the resolution of the camera with the given camera_id. Uses OpenCV to get the resolution.

    Args:
        camera_id (int): The camera ID to get the resolution of. Defaults to 0.

    Returns:
        tuple: A tuple containing the width and height of the camera.
    '''
    cap = cv2.VideoCapture(camera_id)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)

def get_razer_kiyo_resolution() -> tuple:
    '''
    Get the resolution of the Razer Kiyo camera. The resolution is 864x480.

    Returns:
        tuple: A tuple containing the width and height of the Razer Kiyo camera.
    '''
    return (864, 480)