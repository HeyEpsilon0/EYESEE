import cv2      
import time
import win10toast

starttime = time.time()                                                                 #Start time 00
toast=win10toast.ToastNotifier()                                                        #Initialize a notifiaction

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

blink=0

cap = cv2.VideoCapture(0)

ret, image = cap.read()

while ret:

    ret, image = cap.read()                                                             #Start cam
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                #Convert to Grayscale    
    gray_scale = cv2.bilateralFilter(gray_scale, 5, 1, 1)                               #Apply Bilateral Filter
    faces = face_cascade.detectMultiScale(gray_scale, 1.3, 5, minSize=(200, 200))       #Detect faces with Haar Cascades, returns the coordinates of the box containing face.
    if len(faces) > 0:  
        for (x, y, w, h) in faces:                                      

            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)        #Show box over face(if needed)

            eye_face = gray_scale[y:y + h, x:x + w]

            eyes = eyes_cascade.detectMultiScale(eye_face, 1.3, 5, minSize=(50, 50))    #Detect eyes

            if len(eyes) == 2:                                                          #If 2 eyes detected, eyes are open
                

                    cv2.putText(image, "Eyes Open", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)

            elif len(eyes) ==0:                                                         #If less than 2 eyes, it means blink.

                    cv2.putText(image, "Blink ", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)

                    cv2.imshow('EYESEE',image)

                    cv2.waitKey(1)
                    blink = blink + 1                                                   #Increment blink for that 60 seconds
    else:
        cv2.imshow("EYESEE",image)
    a = cv2.waitKey(1)                                                                  #Wait for 1ms 
    if a == 27:                                                                         #If user press Esc. key, terminate
        break

    totaltime = round(time.time() - starttime)   
#Gives time elapsed since start time
    if totaltime%15==0:
        print(totaltime)
        if blink<100:      
            toast.show_toast("EYESEE","You've not been blinking enough!")            #Notify
        print(blink)
        blink=0

print(blink)
print("COMPLETE!")

cap.release()

cv2.destroyAllWindows()
