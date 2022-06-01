import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

WIDTH, HEIGHT = 800, 600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold


def scan_detection(image):
    global document_contour

    document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)


while True:

    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_copy = frame.copy()

    scan_detection(frame_copy)

    cv2.imshow("input", frame)

    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    cv2.imshow("Warped", warped)

    processed = image_processing(warped)
    processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
    cv2.imshow("Processed", processed)

    ocr_text = pytesseract.image_to_string(warped)
    # print(ocr_text)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
