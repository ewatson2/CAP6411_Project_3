import cv2

camera_src = 'http://192.168.1.4:5000'

if __name__ == '__main__':
    camera = cv2.VideoCapture(camera_src)
    
    while True:
        success, frame = camera.read()
        if success:
            cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    camera.release()
