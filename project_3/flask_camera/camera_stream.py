from flask import Flask, Response
from camera import Camera
import cv2

host_addr = '0.0.0.0'
host_port = 5000

camera_src = 0
camera_res = (1280, 720)
camera_fps = 30

app = Flask(__name__)
camera = Camera(camera_src, camera_res, camera_fps).start()

def get_camera_frames():
    while True:
        success, frame = camera.read()
        if success:
            ret, buffer = cv2.imencode('.jpg', frame)
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n'
                  + buffer.tobytes() + b'\r\n')
        else:
            continue

@app.route('/')
def camera_stream():
    return Response(get_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host=host_addr, port=host_port, threaded=True)
    camera.stop()
