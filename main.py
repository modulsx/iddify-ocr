import os
import gradio as gr
from deepface import DeepFace
# from doctr.predictor_models import ocr_predictor
from doctr.io import DocumentFile
from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor
from llm import extract_from_ocr_text

os.environ["USE_TORCH"] = '1'
forward_device = "cpu"

# predictor_model = ocr_predictor(det_arch='linknet_resnet50', reco_arch='vitstr_base', pretrained=True, assume_straight_pages=True)
# predictor_model.det_predictor.predictor_model.postprocessor.bin_thresh = 0.3

predictor_model = load_predictor(det_arch='linknet_resnet50', reco_arch='vitstr_base', assume_straight_pages=True, straighten_pages=False, bin_thresh=0.7, device="cpu")

deepface_backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

def process(image1, image2):
    output1 = DeepFace.extract_faces(img_path = image1, detector_backend = deepface_backends[4])
    output2 = DeepFace.extract_faces(img_path = image2, detector_backend = deepface_backends[4])
    result = DeepFace.verify(img1_path = image1, img2_path = image2)
    ocrImage1 = DocumentFile.from_images(image1)
    ocrImage2 = DocumentFile.from_images(image2)
    ocrResult1 = predictor_model(ocrImage1)
    ocrResult2 = predictor_model(ocrImage2)
    ocrOutput1 = ocrResult1.render()
    ocrOutput2 = ocrResult2.render()
    # llmOutput1 = extract_from_ocr_text(ocrOutput1)
    # llmOutput2 = extract_from_ocr_text(ocrOutput2)
    # extracted_info_1 = [[k,v] for k, v  in llmOutput1.items()]
    # extracted_info_2 = [[k,v] for k, v  in llmOutput2.items()]
    return output1[0]['face'], output2[0]['face'], ocrOutput1, ocrOutput2
    #,extracted_info_1, extracted_info_2, [["Face Matched", result["verified"]]]

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Iddify Demo
    Upload user identifications to verify
    """)
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                inputImage1 = gr.Image(label="ID 1", type="filepath", interactive=True)
                inputImage2 = gr.Image(label="ID 2", type="filepath", interactive=True)
            with gr.Row():
                outImage1 = gr.Image(label="ID 1 Face", type="numpy")
                outImage2 = gr.Image(label="ID 2 Face", type="numpy")
            btn = gr.Button("Verify Documents")
        with gr.Column(scale=1):
            with gr.Row():
                ocrResult1 = gr.Textbox(label="ID 1 OCR")
                ocrResult2 = gr.Textbox(label="ID 2 OCR")
            with gr.Row():
                llmResult1 = gr.Dataframe(label="ID 1 Data", headers=["Label", "Value"])
                llmResult2 = gr.Dataframe(label="ID 2 Data", headers=["Label", "Value"])
            outputTable = gr.Dataframe(label="Output Results", headers=["Parameter", "Result"])
            btn.click(fn=process, inputs=[inputImage1, inputImage2], outputs=[outImage1, outImage2, ocrResult1, ocrResult2, llmResult1, llmResult2, outputTable])

if __name__ == "__main__":
    demo.launch()