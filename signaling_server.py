from flask import Flask, Response, render_template, jsonify, request
import cv2
from pyueye import ueye
import numpy as np
from flask_cors import CORS
import os
import base64
from skimage.metrics import structural_similarity as ssim
import time

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


def capture_frame(frame, num_captures=4):
    captured_files = []
    for i in range(num_captures):
        filename = os.path.join(save_directory, f"captured_{i}.png")
        cv2.imwrite(filename, frame)
        captured_files.append(filename)
        cv2.waitKey(100)
    return captured_files


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


def find_defect(master, images):
    ssim_values = [0, 0, 0, 0]
    classes = [0, 0, 0, 0]
    differences = [None, None, None, None]
    for i, img_path in enumerate(images):
        input_path = img_path
        input = cv2.imread(input_path)
        input = cv2.resize(input, (master.shape[1], master.shape[0]))
        difference, aligned_image, mask_contours, mask_master = align_images(
            master, input
        )
        _, thresholded_diff = cv2.threshold(
            difference, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        mask = cv2.imread("diff_ref.png", cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (master.shape[1], master.shape[0]))
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        diff2 = cv2.absdiff(mask, thresholded_diff)
        similarity_score = ssim(diff2, mask)
        ssim_values[i] = similarity_score
        differences[i] = diff2
        print(similarity_score)
        if similarity_score > 0.85:
            contours, _ = cv2.findContours(
                thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            highlighted_image = aligned_image.copy()
            cnt = 0
            for contour in contours:
                found = False
                for c in mask_contours:
                    if cv2.matchShapes(contour, c, 1, 0.0) < 15:
                        found = True
                        break
                if found:
                    cv2.drawContours(highlighted_image, [contour], -1, (0, 0, 255), 2)
                    cnt += 1
                    classes[i] = 1
    if classes.count(1) > 2:
        return None, True
    idx_arr = [ssim_values[i] for i in range(4) if classes[i] == 0]
    print(ssim_values)
    print(idx_arr)
    print(classes)
    max_idx = ssim_values.index(max(idx_arr))
    return differences[max_idx], False


def capture_distinct_frames(num_frames=4, min_delay=0.5):
    frames = []
    for i in range(num_frames):
        # Clear buffer
        ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
        time.sleep(min_delay)  # Delay between captures

        # Capture new frame
        image_buffer = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        ret = ueye.is_FreezeVideo(hCam, ueye.IS_WAIT)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"Frame capture failed: {ret}")

        # Copy to buffer
        ueye.is_CopyImageMem(hCam, mem_ptr, mem_id, image_buffer.ctypes.data)

        frames.append(image_buffer.copy())
        filename = f"frame_{i}.png"
        cv2.imwrite(os.path.join(save_directory, filename), image_buffer)

    return [os.path.join(save_directory, f"frame_{i}.png") for i in range(num_frames)]


@app.route("/capture", methods=["POST"])
def capture():

    captured_images = capture_distinct_frames()
    if len(captured_images) < 4:
        return jsonify({"error": "Could not capture enough distinct frames"}), 500
    master = request.files["master"]
    master = cv2.imdecode(np.frombuffer(master.read(), np.uint8), cv2.IMREAD_COLOR)
    highlighted_image, res = find_defect(master, captured_images)
    if highlighted_image is None:
        return jsonify({"res": res})
    cv2.imwrite("highlighted_image.png", highlighted_image)
    img = cv2.imread("highlighted_image.png")
    _, img_encoded = cv2.imencode(".png", img)
    highlighted_image_data_url = (
        f"data:image/png;base64,{base64.b64encode(img_encoded).decode('utf-8')}"
    )
    return jsonify({"highlighted_image": highlighted_image_data_url, "res": res})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
