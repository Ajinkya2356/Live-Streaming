from flask import Flask, Response, render_template, jsonify, request
import cv2
from pyueye import ueye
import numpy as np
from flask_cors import CORS
import os
import base64

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

max_width = int(sInfo.nMaxWidth)
max_height = int(sInfo.nMaxHeight)
rectAOI.s32X = ueye.int(0)
rectAOI.s32Y = ueye.int(0)
rectAOI.s32Width = ueye.int(max_width)
rectAOI.s32Height = ueye.int(max_height)
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
        if not ret:
            print("Failed to encode frame to JPEG")
            continue
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


save_directory = "./section_2_clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


def capture_frame(frame):
    roi_image = frame
    filename = os.path.join(save_directory, f"captured.png")
    cv2.imwrite(filename, roi_image)
    print(f"Captured and saved: {filename}")
    return filename


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


""" capture and check simulatenously """


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def align_images(master, input):
    master_preprocessed = preprocess_image(master)
    input_preprocessed = preprocess_image(input)
    sift = cv2.SIFT_create(nfeatures=10000)
    keypoints1, descriptors1 = sift.detectAndCompute(master_preprocessed, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_preprocessed, None)
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    matches_image = cv2.drawMatches(
        master_preprocessed,
        keypoints1,
        input_preprocessed,
        keypoints2,
        good_matches[:50],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite("keypoint_matches.png", matches_image)
    if len(good_matches) >= 4:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )

        # Use RANSAC with refined parameters
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 2.0)

        # ...rest of your existing alignment code...
        aligned_image = cv2.warpPerspective(
            input, H, (master.shape[1], master.shape[0])
        )
        _, mask_master = cv2.threshold(
            master_preprocessed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        mask_contours, _ = cv2.findContours(
            mask_master, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        aligned_image_gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
        aligned_thresh = cv2.threshold(
            aligned_image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        result = cv2.bitwise_or(mask_master, aligned_thresh)
        result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        return result, aligned_image, mask_contours, mask_master
    else:
        raise ValueError("Not enough good matches found for alignment")

def find_defect(master, img_path):
    input_path = img_path
    input = cv2.imread(input_path)
    input = cv2.resize(input, (master.shape[1], master.shape[0]))
    difference, aligned_image, mask_contours, mask_master = align_images(master, input)
    cv2.imwrite("difference.png", difference)
    _, thresholded_diff = cv2.threshold(
        difference, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cv2.imwrite("thresholded_diff.png", thresholded_diff)
    contours, _ = cv2.findContours(
        thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    highlighted_image = aligned_image.copy()
    for contour in contours:
        found = False
        for c in mask_contours:
            if cv2.matchShapes(contour, c, 1, 0.0) < 15:
                found = True
                break
        if found:
            cv2.drawContours(highlighted_image, [contour], -1, (0, 0, 255), 2)
    return highlighted_image


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

    # Capture predefined sections
    captured_images = capture_frame(image_buffer)

    # Check for defects
    master = request.files["master"]
    master = cv2.imdecode(np.frombuffer(master.read(), np.uint8), cv2.IMREAD_COLOR)
    highlighted_image = find_defect(master, captured_images)
    cv2.imwrite("highlighted_image.png", highlighted_image)
    img = cv2.imread("highlighted_image.png")
    _, img_encoded = cv2.imencode(".png", img)
    highlighted_image_data_url = (
        f"data:image/png;base64,{base64.b64encode(img_encoded).decode('utf-8')}"
    )
    return jsonify({"highlighted_image": highlighted_image_data_url})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
