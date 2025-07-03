import cv2 as cv


video_capture = cv.VideoCapture(0)

while True:

    isTrue,frames = video_capture.read()
    # print(frames)
    # height, width, ch =  frames.shape
    grey_frame = cv.cvtColor(frames,cv.COLOR_BGR2GRAY)

    pixel = frames[0][1][1]
    print(pixel)
    h,w = grey_frame.shape
    # print(h)
    if not isTrue:
        print("Error: Couldn't read the frame")
        break
    cv.imshow("video",frames)

    if cv.waitKey(10) & 0xFF == ord('d'):
        break
    
video_capture.release()
cv.destroyAllWindows()