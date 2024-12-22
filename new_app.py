from flask import Flask, Response, render_template, jsonify, request
import cv2
from pyueye import ueye
import numpy as np
from flask_cors import CORS
import os
import base64
from skimage.metrics import structural_similarity as ssim
import time


class CameraError(Exception):
    pass


class ImageProcessingError(Exception):
    pass


class AlignmentError(Exception):
    pass


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})


@app.errorhandler(CameraError)
def handle_camera_error(error):
    return jsonify({"error": str(error)}), 500


@app.errorhandler(ImageProcessingError)
def handle_image_processing_error(error):
    return jsonify({"error": str(error)}), 422


@app.errorhandler(AlignmentError)
def handle_alignment_error(error):
    return jsonify({"error": str(error)}), 422


@app.errorhandler(Exception)
def handle_generic_error(error):
    return jsonify({"error": "Internal server error"}), 500


hCam = ueye.HIDS(0)
sInfo = ueye.SENSORINFO()
cInfo = ueye.CAMINFO()
rectAOI = ueye.IS_RECT()

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

save_directory = "./section_2_clear"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

DELAY_FRAMES = 0.04
NO_FRAMES = 3


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


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def align_images(master, input):
    try:
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
        if len(good_matches) < 4:
            raise AlignmentError("Not enough good matches")
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 2.0)
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
    except Exception as e:
        raise AlignmentError(f"Image alignment failed: {str(e)}")


def find_defect(master, images):
    try:
        ssim_values = [0] * NO_FRAMES
        classes = [0] * NO_FRAMES
        differences = [None] * NO_FRAMES
        print("DETECTION FUNCTION")
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
                classes[i] = 1
        print(classes)
        if classes.count(1) > 2:
            max_ssim = [ssim_values[i] for i in range(NO_FRAMES) if classes[i] == 1]
            max_idx = ssim_values.index(max(max_ssim))
            captured_correct = cv2.imread(images[max_idx])
            return captured_correct, "Pass"
        idx_arr = [ssim_values[i] for i in range(NO_FRAMES) if classes[i] == 0]
        max_idx = ssim_values.index(max(idx_arr))
        captured_incorrect = cv2.imread(images[max_idx])
        return captured_incorrect, "Fail"
    except Exception as e:
        raise ImageProcessingError(f"Defect detection failed: {str(e)}")


def capture_distinct_frames(num_frames=3, min_delay=0.5):
    try:
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

        return [
            os.path.join(save_directory, f"frame_{i}.png") for i in range(num_frames)
        ]
    except Exception as e:
        raise Exception("Not enought images captured")


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
    try:
        captured_images = capture_distinct_frames(
            num_frames=NO_FRAMES, min_delay=DELAY_FRAMES
        )
        if len(captured_images) < NO_FRAMES:
            return jsonify({"error": "Could not capture enough distinct frames"}), 500
        if "master" not in request.form:
            return jsonify({"error": "Master image not provided"}), 400
        master_data_url = request.form["master"]
        header, encoded = master_data_url.split(",", 1)
        master_data = base64.b64decode(encoded)
        master_path = "fetched_master.png"
        with open(master_path, "wb") as f:
            f.write(master_data)
        master = cv2.imread(master_path)
        image, res = find_defect(master, captured_images)
        _, buffer = cv2.imencode(".png", image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "image": f"data:image/png;base64,{image_base64}",
                "res": res,
            }
        )
    except (CameraError, ImageProcessingError, AlignmentError) as e:
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
