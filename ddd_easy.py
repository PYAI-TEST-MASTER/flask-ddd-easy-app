import os
import cv2
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


#classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
INTERMEDIATE_FOLODER = "intermediates"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.h5')#学習済みモデルをロード
#カスケード型分類器に使用する分類器のデータ（xmlファイル）を読み込み
HAAR_FILE = R"./haarcascade_eye_tree_eyeglasses.xml"
cascade = cv2.CascadeClassifier(HAAR_FILE)

@app.route('/', methods=['GET', 'POST'])
# 画像を一枚受け取り、OPEN_EYEかCLOSE_EYEを判定して返す関数
# def pred_gender(img):
#     img = cv2.resize(img, (82,82))
#     img = img.astype('float32') / 255
#     img = np.expand_dims(img, axis=0)
#     pred = model.predict(img)
#     if np.argmax(pred) == 0:
#         return 'OPEN_EYE'
#     else:
#         return 'CLOSE_EYE'



# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('ファイルがありません')
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash('ファイルがありません')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(UPLOAD_FOLDER, filename))
#             filepath = os.path.join(UPLOAD_FOLDER, filename)

#             #受け取った画像を読み込み、np形式に変換
#             img = cv2.imread(filepath)
#             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img_rgb, (82,82))
#             img = img.astype('float32') / 255
#             img = np.expand_dims(img, axis=0)
#             pred = model.predict(img)
#             if np.argmax(pred) == 0:
#                 result = 'OPEN_EYE'    
#             else:
#                 result = 'CLOSE_EYE'

# #            plt.imshow(img)
# #            plt.show()
# #            print(pred_gender(img)) 
# #            img = image.img_to_array(img)
# #            data = np.array([img])
# #            #変換したデータをモデルに渡して予測する
# #            result = model.predict(data)[0]
# #            predicted = result.argmax()
#             pred_answer = "これは " + result + " です"

#             return render_template("index.html",answer=pred_answer)

#     return render_template("index.html",answer="")
def upload_file():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['files']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        files = request.files.getlist('files')  # 'files'はinputタグのname属性
        for file in files:
            # ここでファイルを処理する
            # 例: ファイルを保存する、画像解析を行うなど
            print(file)
            print(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            img = cv2.imread(filepath,0)

            eye = cascade.detectMultiScale(img)
 
            #顔の座標を表示する

            if eye is None:
                print('CLOSE_EYE')
                continue
            try:
                x,y,w,h = eye[0]
            except IndexError:
                print('CLOSE_EYE')
                continue      
            # if eye[0] is None:
            #     print(eye[1])
            #     x,y,w,h = eye[1]
            # else:
            #     print(eye[0])
            #     x,y,w,h = eye[0]
            #顔部分を切り取る

            eye_cut = img[y-h//3:y+h*10//8, x-w//3:x+w*10//8]
#            eye_cut = img[y:y+h, x:x+w]
#            eye_cut = img[y-w//2:y+w//2, x:x+w]
            #白枠で顔を囲む
#            x,y,w,h = eye[0]
            cv2.rectangle(img,(x-w//2,y-h//2),(x+w*10//8,y+h*10//8),(255,255,255),2)
 
            #cv2.rectangle(img,(x,y-w//2),(x+w,y+w//2),(255,255,255),2)
 
            #画像の出力
            filepath = "eye_"+file.filename
            filepath = os.path.join(INTERMEDIATE_FOLODER, filepath)
            cv2.imwrite(filepath, eye_cut)
            filepath = 'face_'+file.filename
            filepath = os.path.join(INTERMEDIATE_FOLODER, filepath)
            cv2.imwrite(filepath, img)
         
            img_rgb = cv2.cvtColor(eye_cut, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img_rgb, (82,82))
            img = img.astype('float32') / 255
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)
            if np.argmax(pred) == 0:
                result = 'OPEN_EYE'    

            else:
                result = 'CLOSE_EYE'
            pred_answer = file.filename+ "は、" + result
            print(pred_answer)
        return render_template("index.html",answer="ファイルが正常にアップロードされました。") 
    return render_template("index.html",answer="")
# if __name__ == "__main__":
#     app.run()
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)