from threading import Thread
import cv2

class Camera:
    def __init__(self, camera_src=0, camera_res=(640, 480), camera_fps=30):
        self.camera = cv2.VideoCapture(camera_src)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_res[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_res[1])
        self.camera.set(cv2.CAP_PROP_FPS, camera_fps)
        
        (self.success, self.frame) = self.camera.read()
        self.stop = False
    
    def start(self):
        t = Thread(target=self.update, name='Camera', args=())
        t.daemon = True
        t.start()
        
        return self
    
    def update(self):
        while True:
            (self.success, self.frame) = self.camera.read()
            if self.stop:
                break
        
        return
    
    def read(self):
        return (self.success, self.frame)
    
    def stop(self):
        self.stop = True
    