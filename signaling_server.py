from flask import Flask, Response, render_template, jsonify
import cv2
from pyueye import ueye
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)

# Initialize the camera
hCam = ueye.HIDS(0)
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
rectAOI = ueye.IS_RECT()

# Initialize the camera
ret = ueye.is_InitCamera(hCam, None)
if ret != ueye.IS_SUCCESS:
    raise Exception(f"Camera initialization failed with error code: {ret}")

# Get camera information
ueye.is_GetCameraInfo(hCam, cInfo)
ueye.is_GetSensorInfo(hCam, sInfo)

# Set color mode to RGB8
ueye.is_SetColorMode(hCam, ueye.IS_CM_BGR8_PACKED)

# Set the area of interest (AOI)
rectAOI.s32X = ueye.int(0)
rectAOI.s32Y = ueye.int(0)
rectAOI.s32Width = ueye.int(1000)  # Set the desired width
rectAOI.s32Height = ueye.int(940)  # Set the desired height
ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_SET_AOI, rectAOI, ueye.sizeof(rectAOI))

camera_width = int(rectAOI.s32Width)
camera_height = int(rectAOI.s32Height)
bitspixel = 24  # for color mode: IS_CM_BGR8_PACKED
mem_ptr = ueye.c_mem_p()
mem_id = ueye.int()

# Allocate memory for the image
ueye.is_AllocImageMem(hCam, camera_width, camera_height, bitspixel, mem_ptr, mem_id)
ueye.is_SetImageMem(hCam, mem_ptr, mem_id)

# Start video capture
ueye.is_CaptureVideo(hCam, ueye.IS_WAIT)


def generate_frames():
    while True:
        # Create the image buffer for capturing frames
        image_buffer = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

        # Capture an image frame
        ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)

        # Copy image data to the buffer
        ueye.is_CopyImageMem(hCam, mem_ptr, mem_id, image_buffer.ctypes.data)

        # Encode image to JPEG format
        ret, buffer = cv2.imencode(".jpg", image_buffer)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


save_directory = "./section_2_clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

predefined_sections = [
    (19, 35, 123, 343),
    (16, 342, 449, 411),
    (119, 5, 441, 343),
]


def capture_predefined_sections(frame, sections):
    captured_images = []
    for i, (x_start, y_start, x_end, y_end) in enumerate(sections):
        roi_image = frame[y_start:y_end, x_start:x_end]
        filename = os.path.join(save_directory, f"section_image_{i}.png")
        cv2.imwrite(filename, roi_image)
        captured_images.append(filename)
        print(f"Captured and saved: {filename}")
    return captured_images


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture", methods=["POST"])
def capture():
    # Create the image buffer for capturing frames
    image_buffer = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

    # Capture an image frame
    ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
    if ret != ueye.IS_SUCCESS:
        return (
            jsonify({"error": f"Failed to capture frame with error code: {ret}"}),
            500,
        )

    # Copy image data to the buffer
    ueye.is_CopyImageMem(hCam, mem_ptr, mem_id, image_buffer.ctypes.data)

    # Resize the captured frame to the desired dimensions
    resized_frame = cv2.resize(image_buffer, (1000, 940))

    # Capture predefined sections
    captured_images = capture_predefined_sections(resized_frame, predefined_sections)

    return jsonify({"captured_images": captured_images})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
