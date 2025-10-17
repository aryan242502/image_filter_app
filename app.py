from flask import Flask, render_template, request, session
import cv2
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filter_choice = request.form.get('filter_choice')

        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['last_image'] = file.filename
        elif 'last_image' in session:
            file = session['last_image']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
        else:
            return render_template('index.html', message="Please upload an image first!")

        img = cv2.imread(filepath)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')

        if filter_choice == 'grayscale':
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif filter_choice == 'blur':
            processed = cv2.blur(img, (10,10))
        elif filter_choice == 'canny':
            processed = cv2.Canny(img, 100, 200)
        elif filter_choice == 'sepia':
            kernel = np.array([[0.272,0.534,0.131],
                               [0.349,0.686,0.168],
                               [0.393,0.769,0.189]])
            processed = cv2.transform(img, kernel)
            processed = np.clip(processed, 0, 255).astype(np.uint8)

        if len(processed.shape) == 2:
            cv2.imwrite(output_path, processed)
        else:
            cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

        return render_template('index.html', original_image=session['last_image'], output_image='output.jpg')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
