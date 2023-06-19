import os
import cv2
from mtcnn import MTCNN

mtcnn = MTCNN()


def crop_faces(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for subdir in os.listdir(input_path):
        subdir_path = os.path.join(input_path, subdir)
        output_subdir_path = os.path.join(output_path, subdir)

        if os.path.isdir(subdir_path):
            if not os.path.exists(output_subdir_path):
                os.makedirs(output_subdir_path)

            for img_file in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_file)
                output_img_path = os.path.join(output_subdir_path, img_file)

                if os.path.isfile(img_path) and not os.path.exists(output_img_path):
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # If MediaPipe fails, try MTCNN
                    print(f"Trying MTCNN for {img_path}")
                    faces = mtcnn.detect_faces(img)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]['box']
                    else:
                        print(f"No face detected in {img_path}")
                        x = y = 0
                        h, w, _ = img.shape

                    cropped_face = img[y:y+h, x:x+w]
                    cv2.imwrite(output_img_path, cropped_face)


# Example usage:
input_path = 'data'
output_path = 'data_crop'
crop_faces(input_path, output_path)
