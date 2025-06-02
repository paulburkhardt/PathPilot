import numpy as np
from typing import Dict, Any
from data_extraction import VideoToFrame, MaSt3RProcessor, SAMProcessor, ObjectCompression, ObjectSegmentation,VLMProcessor
from data_retrieval import DetectionMechanism, WarningGenerator, TextToSpeech
from data_base import DataBase


class Pipelines:

    def __init__(self,config: Dict[str, Any]):
        self.init_pipelines(config)


    def init_pipelines(self,config):
        self.project_stage = config['project_stage']

        #data base
        self.data_base     = DataBase(self.project_stage)

        #data extraction
        self.video_processor        = VideoToFrame(config['video_processor'])
        self.mast3r_processor       = MaSt3RProcessor(config['mast3r'])

        match self.project_stage: # depending on the project stage, we need to initialize different components
            case 1:
                pass
            case 2:
                self.object_segmentation    = ObjectSegmentation(config['object_segmentation'])
                self.object_compression     = ObjectCompression(config['object_compression'])
            case 3:
                self.object_segmentation    = ObjectSegmentation(config['object_segmentation'])
                self.object_compression     = ObjectCompression(config['object_compression'])
                self.vlm_processor          = VLMProcessor(config['vlm_processor'])
        
        #data retrieval
        self.detection_mechanism    = DetectionMechanism(config['detection_mechanism'])
        self.warning_system         = WarningGenerator(config['warning_system'])
        self.text_to_speech         = TextToSpeech(config['text_to_speech'])

    def pipeline_retrieval(self):
        camera_pose = self.mast3r_processor.get_current_camera_pose()
        objects = self.data_base.get_data(camera_pose)

        #retrieval
        objects = self.detection_mechanism.detect_objects()
        self.warning_system.generate_warnings(objects)
        self.text_to_speech.generate_speech(objects)



    def pipeline_extraction(self): 
        
        match self.project_stage:
            case 1:
                self.pipeline_extraction_stage_1()
            case 2:
                self.pipeline_extraction_stage_2()
            case 3:
                self.pipeline_extraction_stage_3()


    def pipeline_extraction_stage_1(self):
        pass


    def pipeline_extraction_stage_2(self):
        pass


    def pipeline_extraction_stage_3(self):
        pass
