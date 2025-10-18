from flask import Flask, render_template, request, session
import cv2
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MAX_DIM = 800

def resize_image(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    blur_level = 10
    canny_low = 50
    canny_high = 150
    sepia_level = 1.0

    if request.method == 'POST':
        filter_choice = request.form.get('filter_choice')
        blur_level = int(request.form.get('blur_level', 10))
        canny_low = int(request.form.get('canny_low', 50))
        canny_high = int(request.form.get('canny_high', 150))
        sepia_level = float(request.form.get('sepia_level', 1.0))

        uploaded_file = request.files.get('image')
        if uploaded_file and uploaded_file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(filepath)
            session['last_image'] = uploaded_file.filename
        elif 'last_image' in session:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['last_image'])
        else:
            return render_template('index.html', message="Please upload an image first!",
                                   blur_level=blur_level, canny_low=canny_low,
                                   canny_high=canny_high, sepia_level=sepia_level)

        img = cv2.imread(filepath)
        img = resize_image(img)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

        if filter_choice == 'grayscale':
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif filter_choice == 'blur':
            processed = cv2.blur(img, (blur_level, blur_level))
        elif filter_choice == 'canny':
            processed = cv2.Canny(img, canny_low, canny_high)
        elif filter_choice == 'sepia':
            kernel = np.array([[0.272*sepia_level,0.534*sepia_level,0.131*sepia_level],
                               [0.349*sepia_level,0.686*sepia_level,0.168*sepia_level],
                               [0.393*sepia_level,0.769*sepia_level,0.189*sepia_level]])
            processed = cv2.transform(img, kernel)
            processed = np.clip(processed,0,255).astype(np.uint8)
        else:
            processed = img

        if len(processed.shape) == 2:
            cv2.imwrite(output_path, processed)
        else:
            cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        return render_template('index.html', original_image=session['last_image'], output_image='output.jpg',
                               blur_level=blur_level, canny_low=canny_low,
                               canny_high=canny_high, sepia_level=sepia_level,
                               filter_choice=filter_choice)

    return render_template('index.html', blur_level=blur_level, canny_low=canny_low,
                           canny_high=canny_high, sepia_level=sepia_level)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
