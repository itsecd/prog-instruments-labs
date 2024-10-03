import sqlite3

import cv2
import numpy as np
from PIL import Image

import dlib_tf_project.utils.tensorflow.style_transfer
from deepface_project.model_test import train_classifier
from deepface_project.utils.get_embeddings import get_embedding

QUERY = """select value from embedding where info LIKE ? """


def get_preprocessed_data(db_data):
    for row in db_data:
        row[0] = row[0][1:-1]
        row[0] = np.fromstring(row[0], dtype=float, sep=', ')
    return db_data


def get_embeds_from_db(cursor):
    str_to_like_operand = "sketch_to_pipeline_%"
    db_data = [list(item) for item in cursor.execute(QUERY, [str_to_like_operand]).fetchall()]
    return get_preprocessed_data(db_data)


def find_in_db(embeds_array, embed, clf, pca):
    embed = np.array([x for x in embed])
    result_array = []
    for embed1 in embeds_array:
        result_array.append(embed1 - embed)
    for i in range(0, len(result_array)):
        result_array[i] = result_array[i].flatten()
    result_array = np.array(result_array)
    data = pca.transform(result_array)
    print('============================')
    tmp = clf.predict_proba(data)
    print(tmp)
    print('============================')
    for i in range(0, len(tmp)):
        if tmp[i, 0] > 0.85:
            print(f'a person with an id {i} was found')



def transfer_style(content_img_path, style_img_path, result_img_path, transfer_model):
    transfer_model.process_image(content_img_path, style_img_path, result_img_path)


if __name__ == "__main__":

    try:
        transfer_model = dlib_tf_project.utils.tensorflow.style_transfer.TransferModel(
            dlib_tf_project.utils.tensorflow.style_transfer.MODEL_URL)
    except (RuntimeError, TypeError, NameError):
        print('Error getting transfer model')
        exit()

    video_path = "C:\\CompositePortraitRecongnition\\dlib_tf_project\\video\\test2.mp4"
    video_capture = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        "C:\\CompositePortraitRecongnition\\dlib_tf_project\\pipeline_implementation\\haarcascade_frontalface_default.xml")

    clf, pca = train_classifier()
    model_name = "Facenet512"

    style_img_path = "C:\\CompositePortraitRecongnition\\common\\dataset\\sketches_to_pipeline\\2.jpg"

    count = 1

    conn = sqlite3.connect("../../common/db/database.db")
    cursor = conn.cursor()
    embeds_array = get_embeds_from_db(cursor)

    while True:
        ret, img = video_capture.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w + 10, y + h + 10), (255, 255, 0), 2)
            rec_gray = gray_img[y:y + h, x:x + w]
            rec_color = img[y:y + h, x:x + w]
            tmp_path_to_save = f"tmp_{count}.jpg"
            path_to_save_stylized_img = f"stylized_{count}.jpg"
            cv2.imwrite(tmp_path_to_save, gray_img[y:y + h + 10, x:x + w + 10])
            try:
                embed = get_embedding(tmp_path_to_save, model_name)
            except ValueError:
                continue
            portrait_image = Image.open(tmp_path_to_save)
            portrait_image.thumbnail((350, 500))
            portrait_image.save(tmp_path_to_save)
            transfer_style(tmp_path_to_save, style_img_path, path_to_save_stylized_img,
                           transfer_model)
            try:
                embed = get_embedding(path_to_save_stylized_img, model_name)
            except ValueError:
                continue
            find_in_db(embeds_array, embed, clf, pca)

            count += 1

        cv2.imshow('Face Recognition', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()
