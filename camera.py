import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# def classify_sign(frame, contour, shape,count_red,count_blue):
#     x, y, w, h = cv2.boundingRect(contour)

#     sign_roi = frame[y:y+h, x:x+w]  

#     if(shape=="triangle"):
#         template_dir = 'C:/Users/Admin/Downloads/traffic_sign/Triangle'
#     elif(shape=="rectangle") :   
#         template_dir = 'C:/Users/Admin/Downloads/traffic_sign/Rectangle'
#     elif(shape=="circle"):
#         if(count_blue):    
#             template_dir = 'C:/Users/Admin/Downloads/traffic_sign/Circle/Blue'
#         else:
#             template_dir = 'C:/Users/Admin/Downloads/traffic_sign/Circle/Red'
#     max_value = -1
#     ten_file = ""

#     for template_filename in os.listdir(template_dir):
#         template_path = os.path.join(template_dir, template_filename)
        
#         template = cv2.imread(template_path)  

#         if template is None:
#             print(f"Không thể đọc file: {template_filename}")
#             continue

#         sign_roi_resized = cv2.resize(sign_roi, (template.shape[1], template.shape[0]))

#         result = cv2.matchTemplate(sign_roi_resized, template, cv2.TM_CCORR_NORMED)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
#         if max_val > max_value : 
#             max_value = max_val
#             ten_file = template_filename  

#     if max_value > 0.7:  
#         ten_file_name, _ = os.path.splitext(ten_file)  # Tách tên file và phần mở rộng
#         print(f"Biển báo khớp với mẫu: {ten_file_name} với độ tương đồng {max_value}")
#         return ten_file_name


def process_frame(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    v_clahe = clahe.apply(v)

    hsv_clahe = cv2.merge((h, s, v_clahe))

    img_median = cv2.medianBlur(hsv_clahe, 5)
    # Red mask
    lower_red1 = np.array([0, 130, 120])    # Lower bound for red color
    upper_red1 = np.array([10, 255, 255])   # Upper bound for lower red range
    lower_red2 = np.array([160, 130, 120])  # Lower bound for upper red range
    upper_red2 = np.array([180, 255, 255])  # Upper bound for red color


    mask1 = cv2.inRange(img_median, lower_red1, upper_red1)
    mask2 = cv2.inRange(img_median, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)


    kernel = np.ones((7,7), np.uint8)

    red_mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    red_mask_cleaned = cv2.morphologyEx(red_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # blue mask
    lower_blue = np.array([90, 130, 110])
    upper_blue = np.array([120, 255, 255])
    blue_mask=cv2.inRange(img_median,lower_blue,upper_blue)

    kernel = np.ones((5,5), np.uint8)

    blue_mask_cleaned = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    blue_mask_cleaned = cv2.morphologyEx(blue_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    contours_red, hierarchy = cv2.findContours(red_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, hierarchy_blue = cv2.findContours(blue_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count_red = 1 if contours_red else 0
    count_blue = 1 if contours_blue else 0

    contours_combined = contours_red + contours_blue
    contour_image = frame.copy()

    for cnt in contours_combined:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if area > 2000:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if len(approx) == 3:  # Tam giác
                cv2.drawContours(contour_image, [cnt], 0, (0, 255, 0), 2)
                cv2.putText(contour_image,classify_sign(contour_image, cnt, "triangle",count_red,count_blue), (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 255, 0), 2)
            
            elif len(approx) == 4:  # Hình chữ nhật
                cv2.drawContours(contour_image, [cnt], 0, (0, 255, 0), 2)
                cv2.putText(contour_image,classify_sign(contour_image, cnt, "rectangle",count_red,count_blue), (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            elif len(approx) > 6 and len(approx) <10:  # Hình tròn
                cv2.drawContours(contour_image, [cnt], 0, (0, 255, 0), 2)
                cv2.putText(contour_image,classify_sign(contour_image, cnt, "circle",count_red,count_blue), (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 , 0, 255), 2)
    return contour_image

# Khởi động camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể lấy khung hình từ camera.")
        break
    red_result = process_frame(frame)
        
    cv2.imshow("Traffic Signs Detection and Classification", red_result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
