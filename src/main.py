import cv2
import pandas as pd
import numpy as np
import math
from ultralytics import YOLO
import csv
from sort import*
path = os.path.expanduser("~/Desktop/c.csv")


# field names
fields = ["Type",
    "Vehicle ID.",
    "Frame Number",
    "Centre X",
    "Centre Y",
    "Real X",
    "Real Y",
    "Time",
]

vedio = "data/input/<filename.extension>"


frame_number = 0        #to define the frame an d frame rate
data=[]                 #to store the final data
pixel_coordinates=[]    #to store the pixel coordinates entered by user

center_x1=0             #to differnciate plane in case of no liner transformation 
center_y1=0
center_x2=0
center_y2=0
line=0

#initating model
model=YOLO('weights.pt')



# function to input the pixel coordinates
def collect_real_coordinates(event, x, y, flags, param ):

    if event == cv2.EVENT_MOUSEMOVE: 

        zoom_x = max(x - 25, 0)
        zoom_y = max(y - 25, 0)

        zoomed_image = frame2[zoom_y:zoom_y + 50, zoom_x:zoom_x + 50]
        zoomed_image = cv2.resize(zoomed_image, (200, 200))
        cv2.circle(zoomed_image,(100,100),4,(0,0,255),-1)
        cv2.imshow('Zoomed In', zoomed_image)

    if event == cv2.EVENT_LBUTTONDOWN & len(pixel_coordinates)<4:
        coordinates = [x, y]
        pixel_coordinates.append(coordinates)
        print("the endtered coordinate is: ", coordinates)

#distance between two points
def euclideanDistance(pt1, pt2):
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), .5)

#plane decinding function
def define_plane(x,y):
    return  (int(center_x2-center_x1)*int(center_y2-y))-(int(center_y2-center_y1)*int(center_x2-x))


# Display the first frame as an image to locate pixel coordinates
cv2.namedWindow('First Frame')
cv2.setMouseCallback('First Frame',collect_real_coordinates )
cap=cv2.VideoCapture(vedio)
ret, frame2 = cap.read()
if ret:
    frame2=cv2.resize(frame2,(1536,864 ))
    cv2.imshow("First Frame", frame2)
    cv2.waitKey(0)
    


cap.release()
cv2.destroyAllWindows()
print(pixel_coordinates)


#input the real coordinates according to the pixel coordinates
print("enter real coordinates according to the corresponding pixel coordinate for first plane")
x1, y1, x2, y2, x3, y3, x4, y4 = input().split()
x1 ,x2 , x3 , x4 = float(x1) , float(x2) , float(x3) , float(x4)
y1 ,y2 ,y3 ,y4 = float(y1), float(y2) , float(y3) ,float(y4)

# #pixel coordinates
pixel_coords_one = np.array([pixel_coordinates[1], pixel_coordinates[3], pixel_coordinates[5], pixel_coordinates[7]])


def euclideanDistance(pt1, pt2):
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), .5)


#real coordinates
real_coords_one = np.array(
    [
        [float(x1), float(y1)],
        [float(x2), float(y2)],
        [float(x3), float(y3)],
        [float(x4), float(y4)],
    ]
)


#homology matrix
H1, _ = cv2.findHomography(pixel_coords_one, real_coords_one)

#if 2nd plane is introducedd
if len(pixel_coordinates)==17:
    for i in range(0,4):
        i=(2*i)+1
        for j in range(4,8):
            j=(2*j)+1
            length= euclideanDistance(pixel_coordinates[i],pixel_coordinates[j])
            if i == 1 and j==9:
                length_min_1=length
                length_min_2=length
                plane_one_first_cordinte=pixel_coordinates[i]
                plane_two_first_coordinate=pixel_coordinates[j]
                plane_one_second_cordinte=pixel_coordinates[i]
                plane_two_second_coordinate=pixel_coordinates[j]
            else:
                if length_min_1>length:
                    length_min_2=length_min_1
                    plane_one_second_cordinte=plane_one_first_cordinte
                    plane_two_second_coordinate=plane_two_first_coordinate
                    length_min_1=length
                    plane_one_first_cordinte=pixel_coordinates[i]
                    plane_two_first_coordinate=pixel_coordinates[j]
                    center_x1 = (plane_one_first_cordinte[0] + plane_two_first_coordinate[0])/2
                    center_y1 = (plane_one_first_cordinte[1] + plane_two_first_coordinate[1])/2
                    center_x2 = (plane_one_second_cordinte[0]+plane_two_second_coordinate[0])/2
                    center_y2 = (plane_one_second_cordinte[1]+plane_two_second_coordinate[1])/2



    print("enter real coordinates according to the corresponding pixel coordinate for 2nd plane")
    x5 , y5 ,x6 , y6 , x7 ,y7 , x8 , y8 = input().split()
    x5 ,x6 , x7 , x8 = float(x5) ,float(x6) , float(x7) , float(x8)
    y5 ,y6 ,y7 ,y8 = float(y5), float(y6) , float(y7) ,float(y8)

    pixel_coords_two = np.array([pixel_coordinates[9], pixel_coordinates[11], pixel_coordinates[13], pixel_coordinates[15]])

    real_coords_two=np.array(
    [
        [float(x5), float(y5)],
        [float(x6), float(y6)],
        [float(x7), float(y7)],
        [float(x8), float(y8)],
    ]
    )

    H2,_=cv2.findHomography(pixel_coords_two,real_coords_two) 
        
    

    line=define_plane(int(pixel_coordinates[1][0]),int(pixel_coordinates[1][1]))

    if line > 0:
        homology_matrix_one = H1
        homology_matrix_two = H2
    else:
        homology_matrix_one=H2
        homology_matrix_two=H1   




#Inetating Sort Tracker
Tracker = Sort(max_age=30 , min_hits=3 , iou_threshold=0.3)

cap=cv2.VideoCapture(vedio)
while True:
    timer = cv2.getTickCount()  
    success , img=cap.read()
    vehicle_type=[]
    if img is not None:
        img =cv2.resize(img,(1536,864))
    # imgr = cv2.bitwise_and(img,mask)
    # result = model(img,show=True)
    
    # img = cv2.resize(img,(1020,500))
    results = model(img,stream=True)
    detections_car= np.empty((0,5))



    collect=[]


    for result in  results:
        boxes = result.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1 , y1 , x2 , y2= int(x1),int(y1),int(x2),int(y2)
            # use to dra rectangle around the object
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,100,0),3)
            cx,cy = (x1+x2)/2 , (y1+y2)/2
            conf = math.ceil(box.conf[0])
            cls = int(box.cls[0])
            if cls == 2 or cls==3 or cls==7 or cls==5:
                dets_prep_car= np.array([x1,y1,x2,y2,conf ])
                detections_car=np.vstack((detections_car ,dets_prep_car))


    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    track_result_car= Tracker.update(detections_car)
    frame_number += 1
    current_frame=frame_number
    video_time_sec = current_frame / fps
    video_time = round(video_time_sec , 3)

    i=0
    for res in track_result_car:
        
        x3 , x4 , y3 , y4 , s  = res 
        cx=(x1+x2)/2
        cy=(y3+y4)/2
        if math.isnan(cx) or math.isnan(cy):
            cx=0
            cy=0
            s=0        
        else:
            cx=int((x3+y3)/2)
            cy=int((x4+y4)/2)
            s=int(s)
        cv2.circle(img,(cx,cy),4,(0,0,255),-1)
        cv2.putText(img,str(s),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,205),2)


        #output
        point_pixel = np.array([[cx, cy]], dtype=np.float32)
        homogeneous_coords = np.hstack([point_pixel, np.ones((1, 1))])
        if line==0:
            homogeneous_coords_transformed = np.dot(H1 , homogeneous_coords.T)
            real_coords_transformed = (
                homogeneous_coords_transformed[:2]
                / homogeneous_coords_transformed[2]
            )
        else:
            if define_plane(cx , cy ) > 0:
                homogeneous_coords_transformed = np.dot(homology_matrix_one, homogeneous_coords.T)
                real_coords_transformed = (
                homogeneous_coords_transformed[:2]
                / homogeneous_coords_transformed[2]
                )
            else:
                homogeneous_coords_transformed = np.dot(homology_matrix_two, homogeneous_coords.T)
                real_coords_transformed = (
                homogeneous_coords_transformed[:2]
                / homogeneous_coords_transformed[2]
                )
        # print(real_coords_transformed[0][0], real_coords_transformed[1][0])


        row = [   
                s,
                cx,
                cy,
                real_coords_transformed[0][0],
                real_coords_transformed[1][0],
                video_time,
            ]
        data.append(row)
        cv2.imshow("RGB", img)
    if cv2.waitKey(1)&0xFF==27:
        with open("results/data.csv","w+",newline='') as data_csv:
            row = ["type","frame" , "id" , "pixel x" , "pixel y" , "real x" , "real y" , "time"]
            csvWriter = csv.writer(data_csv,delimiter=',')
            csvWriter.writerow(row)
            csvWriter.writerows(data)
        break
    else:
        print("the program come to an end")
    


cap.release()
cv2.destroyAllWindows()
