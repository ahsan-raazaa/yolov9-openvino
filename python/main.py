from pathlib import Path

import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.preprocess import ColorFormat
from openvino.runtime import Layout, Type

import numpy as np
import cv2

class Yolov9:
    def __init__(self, xml_model_path="./model/yolov9-c-converted.xml", conf=0.2, nms=0.4):
        # Step 1. Initialize OpenVINO Runtime core
        core = ov.Core()
        # Step 2. Read a model
        model = core.read_model(str(Path(xml_model_path)))

        # Step 3. Inizialize Preprocessing for the model
        ppp = PrePostProcessor(model)
        # Specify input image format
        ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC")).set_color_format(ColorFormat.BGR)
        #  Specify preprocess pipeline to input image without resizing
        ppp.input().preprocess().convert_element_type(Type.f32).convert_color(ColorFormat.RGB).scale([255., 255., 255.])
        # Specify model's input layout
        ppp.input().model().set_layout(Layout("NCHW"))
        #  Specify output results format
        ppp.output().tensor().set_element_type(Type.f32)
        # Embed above steps in the graph
        model = ppp.build()

        self.compiled_model = core.compile_model(model, "CPU")

        self.input_width = 640
        self.input_height = 640
        self.conf_thresh = conf
        self.nms_thresh = nms

    def resize_and_pad(self, image):

        old_size = image.shape[:2] 
        ratio = float(self.input_width/max(old_size))#fix to accept also rectangular images
        new_size = tuple([int(x*ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))
        
        delta_w = self.input_width - new_size[1]
        delta_h = self.input_height - new_size[0]
        
        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
        
        return new_im, delta_w, delta_h

    def predict(self, img):

        # Step 4. Create tensor from image
        input_tensor = np.expand_dims(img, 0)

        # Step 5. Create an infer request for model inference 
        infer_request = self.compiled_model.create_infer_request()
        infer_request.infer({0: input_tensor})

        # Step 6. Retrieve inference results 
        output = infer_request.get_output_tensor()
        detections = output.data[0].T

        # Step 7. Postprocessing including NMS  
        boxes = []
        class_ids = []
        confidences = []
        for prediction in detections:
            classes_scores = prediction[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > self.conf_thresh):
                confidences.append(classes_scores[class_id])
                class_ids.append(class_id)
                x, y, w, h = prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()
                xmin = x - (w / 2)
                ymin = y - (h / 2)
                box = np.array([xmin, ymin, w, h])
                boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)

        detections = []
        for i in indexes:
            j = i.item()
            detections.append({"class_index": class_ids[j], "confidence": confidences[j], "box": boxes[j]})

        return detections

    def draw(self, img, detections, dw, dh):
        # Step 8. Print results and save Figure with detections
        for detection in detections:
        
            box = detection["box"]
            classId = detection["class_index"]
            confidence = detection["confidence"]

            rx = img.shape[1] / (self.input_width - dw)
            ry = img.shape[0] / (self.input_height - dh)
            box[0] = rx * box[0]
            box[1] = ry * box[1]
            box[2] = rx * box[2]
            box[3] = ry * box[3]

            xmax = box[0] + box[2]
            ymax = box[1] + box[3]
            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(xmax), int(ymax)), (0, 255, 0), 3)
            img = cv2.rectangle(img, (int(box[0]), int(box[1]) - 20), (int(xmax), int(box[1])), (0, 255, 0), cv2.FILLED)
            img = cv2.putText(img, str(classId), (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))        

def main( ):

    model = Yolov9("./model/yolov9-c-converted.xml")

    img = cv2.imread("./000000000312.jpg")

    img_resized, dw, dh = model.resize_and_pad(img)
    results = model.predict(img_resized)
    model.draw(img, results, dw, dh)


    cv2.imshow("./detection_python.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main( )
