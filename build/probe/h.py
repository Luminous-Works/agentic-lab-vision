import json
def handler(event, context):
    import cv2, numpy as np
    img = np.zeros((64, 64, 3), np.uint8); cv2.circle(img, (32, 32), 10, (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    back = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return {"cv2": cv2.__version__, "numpy": np.__version__, "decoded_shape": list(back.shape)}
