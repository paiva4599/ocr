import time
start_time = time.time()
import os
import cv2
import pytesseract
import prep
from pyspark import SparkContext
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' 
tessdata_dir_config = '--oem 3 --psm 6 --tessdata-dir "/user/projects/ocr/tessdata"' 
language = 'por+lat'  

sc = SparkContext("ip_do_master", "OCR Preprocessing")

image_folder = "/user/projects/ocr/non-processed"
reference_text_folder = "/user/projects/ocr/reference_texts"

super_res_model_path = "/user/projects/ocr/models/ESPCN_x2.pb" 


def calculate_accuracy(extracted_text, reference_text):
    return SequenceMatcher(None, extracted_text, reference_text).ratio()


def apply_super_resolution(image, model_path):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("espcn", 2)
    upscaled_image = sr.upsample(image)
    return upscaled_image


def preprocess_and_ocr(image_path):
    image = cv2.imread(image_path)
    upscale = apply_super_resolution(image, super_res_model_path)
    cv2.imwrite(f"/user/projects/ocr/processed/1_upscale_{os.path.basename(image_path)}", upscale)
    
    grayscale = prep.get_grayscale(image)
    cv2.imwrite(f"/user/projects/ocr/processed/2_grayscale_{os.path.basename(image_path)}", grayscale)
    
    noise = prep.remove_noise(grayscale)
    cv2.imwrite(f"/user/projects/ocr/processed/2_grayscale_{os.path.basename(image_path)}", noise)
    
    thresh = prep.thresholding(noise)
    cv2.imwrite(f"/user/projects/ocr/processed/3_thresh_{os.path.basename(image_path)}", thresh)
    
    dilate = prep.dilate(thresh)
    cv2.imwrite(f"/user/projects/ocr/processed/4_dilate_{os.path.basename(image_path)}", dilate)
    
    erode = prep.erode(dilate)
    cv2.imwrite(f"/user/projects/ocr/processed/5_erode_{os.path.basename(image_path)}", erode)

    extracted_text = pytesseract.image_to_string(erode, lang=language, config=tessdata_dir_config)
   
    reference_text_path = os.path.join(reference_text_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    if os.path.exists(reference_text_path):
        with open(reference_text_path, 'r', encoding='utf-8') as ref_file:
            reference_text = ref_file.read()
    else:
        reference_text = "" 

    accuracy = calculate_accuracy(extracted_text, reference_text)
   
    return (image_path, extracted_text, accuracy)


image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) \
    if img.endswith(('.png', '.jpg', '.jpeg'))]

images_rdd = sc.parallelize(image_paths)

results = images_rdd.map(preprocess_and_ocr).collect()

for processed_image, extracted_text, accuracy in results:
    print(f"Imagem: {processed_image.split('/')[-1]}\
        \nTexto Extraído:\n{extracted_text}\
        Acurácia: {accuracy:.2f}\n")

acuracias = [result[2] for result in results]
media_acuracia = sum(acuracias) / len(acuracias)
print(f'\nAcurácia Média: {media_acuracia:.2f}')
print(f'Núcleos: {images_rdd.getNumPartitions()}')


execution_time = time.time() - start_time
print(f'Tempo de Execução: {execution_time:.2f}')