import cv2 

cap = cv2.VideoCapture(0)

flag , prev_frame = cap.read()

while True:

    flag, curr_frame= cap.read()

    diff = cv2.absdiff(prev_frame,curr_frame)

    cv2.imshow("diff", diff)

    prev_frame = curr_frame

    if cv2.waitKey(1000) & 0xFF == 27:
       break
cap.release()
cv2.destroyAllWindows()

