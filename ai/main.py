
import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound

def calculate_EAR(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (a+b)/(2.0*c)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0) #this will capturne the video from your default camera

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read() #this will store frames recorded by camera to frame variable
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #here we are making the frame gray to make working with it easy

    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face) #now this will give us landmarks of the face

        #now to detect the eye land mark we can do this
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            nextPoint = n + 1
            if n == 41:
                nextPoint = 36
            x2 = face_landmarks.part(nextPoint).x
            y2 = face_landmarks.part(nextPoint).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)


        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            nextPoint = n + 1
            if n == 47:
                nextPoint = 42
            x2 = face_landmarks.part(nextPoint).x
            y2 = face_landmarks.part(nextPoint).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)


        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2

        EAR = round(EAR, 2)

        if EAR < 0.2:
            cv2.putText(frame, "DROWSY",(20,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
            cv2.putText(frame, "ARE YOU SLEEPY???", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            playsound('beep.wav')
            print("DROWSY")
        print(EAR)

        cv2.imshow("Are you sleepy", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



