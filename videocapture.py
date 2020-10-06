import cv2
#import matplotlib.pyplot as plt

cap = cv2.VideoCapture('filename')
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0
j = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    #for i in range(10):
    #    ret, frame = cap.read()
    if ret == True:
        
        if (i == 50):
            cv2.imwrite('opencv'+str(j)+'.png ', frame)
            j = j + 1
            i = 0
        
        i = i + 1
#    i = 0
#    if ret == True:
#        cv2.imwrite('opencv'+str(i)+'.png ', frame)
#        i = i + 1
        
#         img_counter = 0
#         
#         img_name = "opencv_frame_{}.png".format(img_counter)
#         cv2.imwrite(img_name, frame)
#         print("{} written!".format(img_name))
#         img_counter += 1
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        cv2.imshow('frame', gray)
#
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


