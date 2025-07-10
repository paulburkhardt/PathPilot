# Add the MASt3r-SLAM root directory to sys.path if not already present
import sys
import os
mast3r_slam_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mast3r_slam"))
if mast3r_slam_root not in sys.path:
    sys.path.insert(0, mast3r_slam_root)

import time
from typing import List, Dict, Any, Tuple, Literal
from .abstract_slam_component import AbstractSLAMComponent
import torch.multiprocessing as mp
import torch
import lietorch
import numpy as np
import PIL

from .mast3r_slam_component_utils.shared_keyframes import SharedKeyframes
from .mast3r_slam_component_utils.frame import create_frame


#from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.frame import Mode, SharedStates
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
    resize_img
)
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.geometry import constrain_points_to_ray


from src.pipeline.data_entities.image_data_entity import ImageDataEntity
from src.pipeline.data_entities.point_cloud_data_entity import PointCloudDataEntity


#class Segmented_frame(Frame):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.image_segmentation_mask = None
#
#
#def _resize_pil_seg_mask(img, long_edge_size):
#    S = max(img.size)
#    interp = PIL.Image.NEAREST
#    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
#    return img.resize(new_size, interp)
#
#
#def resize_seg_mask(img, size, square_ok=False, return_transformation=False):
#    assert size == 224 or size == 512
#    # numpy to PIL format
#    img = PIL.Image.fromarray(np.uint8(img))
#    W1, H1 = img.size
#    if size == 224:
#        # resize short side to 224 (then crop)
#        img = _resize_pil_seg_mask(img, round(size * max(W1 / H1, H1 / W1)))
#    else:
#        # resize long side to 512
#        img = _resize_pil_seg_mask(img, size)
#    W, H = img.size
#    cx, cy = W // 2, H // 2
#    if size == 224:
#        half = min(cx, cy)
#        img = img.crop((cx - half, cy - half, cx + half, cy + half))
#    else:
#        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
#        if not (square_ok) and W == H:
#            halfh = 3 * halfw / 4
#        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
#
#    unnormalized_img=np.asarray(img),
#
#    return unnormalized_img
#
#
#def create_frame_with_segmentation_mask(i, img, T_WC, img_size=512, device="cuda:0", image_segmentation_mask=None):
#    """
#    Overwrites the create_frame method of masterslam to add in the segmentation mask
#
#    Args:
#        
#    Returns:
#
#    """
#
#    #----- untouched code from masterslam----------
#    img = resize_img(img, img_size)
#    rgb = img["img"].to(device=device)
#    img_shape = torch.tensor(img["true_shape"], device=device)
#    img_true_shape = img_shape.clone()
#    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
#    downsample = config["dataset"]["img_downsample"]
#    if downsample > 1:
#        uimg = uimg[::downsample, ::downsample]
#        img_shape = img_shape // downsample
#    #frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
#    #-----------------------------------------------
#    frame = Segmented_frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
#
#    if image_segmentation_mask is not None:
#        
#        #repeat the image segmentation mask along the third dimension for compatibility with masterslam
#        image_segmentation_mask = image_segmentation_mask.as_pytorch()
#        H,W = image_segmentation_mask.shape
#        image_segmentation_mask = image_segmentation_mask.unsqueeze(-1).expand(H,W,3)
#        image_segmentation_mask = resize_seg_mask(image_segmentation_mask,img_size)[0]
#        image_segmentation_mask = torch.from_numpy(image_segmentation_mask)
#        image_segmentation_mask = image_segmentation_mask[:,:,0]
#        if downsample >1:
#            image_segmentation_mask = image_segmentation_mask[::downsample,::downsample]
#
#        #collapse the last dim again as they are unnecessary
#        frame.image_segmentation_mask = image_segmentation_mask
#
#    return frame


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    """
    Relocalization method of masterslam

    Args:
        -
    Returns:
        - 
    Raises:
        -
    """
    
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    """
    Backend method of masterslam

    Args:
        -
    Returns:
        -
    Raises:
        -
    """
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model, retriever_path = "MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)




class MAST3RSLAMComponent(AbstractSLAMComponent):
    """
    SLAM component implementing the MAST3R SLAM algorithm.

    Args:
        c_confidence_threshold: Confidence threshold above which points are output.
        mast3r_slam_config_path: Full path to the config of Mast3r Slam to use.
        device (str, optional): The device on which to run MAST3R SLAM. Defaults to "cuda:0".
    """
    
    def __init__(
        self,
        point_cloud_method: Literal["accumulating", "refreshing"],
        c_confidence_threshold: float,
        mast3r_slam_config_path: str, 
        device: str = "cuda:0", 
        segment_point_cloud: bool = True
        )->None:

        super().__init__()

        load_config(mast3r_slam_config_path)
        self.device = device
        self.c_confidence_threshold = c_confidence_threshold
        self.segment_point_cloud = segment_point_cloud

        assert point_cloud_method in ["accumulating", "refreshing"], "point_cloud_method must be either 'accumulating' or 'refreshing'"
        self.point_cloud_method = point_cloud_method

        self._is_inited = False


        #prepare MAST3R Slam init based on their main
        mp.set_start_method("spawn")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)    
        self.manager = mp.Manager()
        

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires RGB images as input."""
    
        inputs = ["image","step_nr","timestamp","image_size","image_width","image_height", "calibration_K"]

        if self.segment_point_cloud:
            inputs.append("image_segmentation_mask")

        return inputs
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs point clouds and camera poses."""
        return ["point_cloud", "camera_pose", "key_frame_flag"]

    def _init_mast3r_slam(
            self,
            img_height:int,
            img_width: int,
            calibration_K = None):
        """
        inits the stuff that is dataset parameter dependent
        """

        self.keyframes = SharedKeyframes(self.manager,img_height, img_width)
        self.states = SharedStates(self.manager, img_height, img_width)

        self.model = load_mast3r(device=self.device,path="MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" )
        self.model.share_memory()

        if calibration_K is not None:
            self.keyframes.set_intrinsics(calibration_K)

        self.tracker = FrameTracker(self.model,self.keyframes,self.device)
        self.backend = mp.Process(target=run_backend, args=(config, self.model, self.states, self.keyframes, calibration_K))
        self.backend.start()


    def __del__(self):
        """
        Terminate the backend of masterslam on exit.

        Args:
            -

        Returns:
            -

        Raises:
            -
        """
        
        if hasattr(self, "backend") and self.backend.is_alive():
            self.backend.terminate()
            self.backend.join()

    def _run(self, 
             image: ImageDataEntity,
             timestamp: Any, 
             step_nr: int,
             image_size:int,
             image_width: int,
             image_height: int,
             calibration_K: Any=None,
             image_segmentation_mask: Any=None) -> Dict[str, Any]:
        """
        Process an RGB image using MAST3R SLAM.

        Args:
            image: The input image.
            timestamp: Timestamp of the image within the video.
            step_nr: Frame number.
            image_size: Size of the image (total number of pixels).
            image_width: Width of the image.
            image_height: Height of the image.
            calibration_K: Camera Calibration matrix
            image_segmentation_mask: Segmentation of the input image

        Returns:
            Dict[str, Any]: A dictionary containing the point cloud ("point_cloud") and camera pose ("camera_pose").
        """
        

        if self.segment_point_cloud:
            assert image_segmentation_mask is not None, "segment_point_cloud is set to True, but no segmentation mask provided."

        if not self._is_inited:
            self._init_mast3r_slam(img_height=image_height,img_width=image_width,calibration_K=calibration_K)
            self._is_inited = True

        mode = self.states.get_mode()
        
        T_WC = (
            lietorch.Sim3.Identity(1, device=self.device)
            if step_nr == 0
            else self.states.get_frame().T_WC
        )

        #frame = create_frame(step_nr, image.as_numpy(), T_WC, img_size=image_size, device=self.device)
        frame = create_frame(
            step_nr, 
            image.as_numpy(), 
            T_WC, 
            img_size=image_size, 
            device=self.device,
            img_segmentation_mask= image_segmentation_mask
        )



        X_init,C_init = None, None
        X,C = None, None

        add_new_kf = False
        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X_init, C_init)
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            self.states.set_mode(Mode.TRACKING)
            self.states.set_frame(frame)
            
        elif mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = self.tracker.track(frame)
            if try_reloc:
                self.states.set_mode(Mode.RELOC)
            self.states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(self.model, frame)
            frame.update_pointmap(X, C)
            self.states.set_frame(frame)
            self.states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with self.states.lock:
                    if self.states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            self.keyframes.append(frame)
            self.states.queue_global_optimization(len(self.keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with self.states.lock:
                    if len(self.states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)






        #project the keyframes onto the current pointcloud
        
        pointclouds = []
        colors = []
        confidence_scores = []
        if self.segment_point_cloud:
            segmentation_masks = []
        else:
            segmentation_masks = None

        # variant 1: update the pointcloud merely with every keyframe
        if self.point_cloud_method == "accumulating":
            for i in range(len(self.keyframes)):
                keyframe = self.keyframes[i]
                if config["use_calib"]:
                    X_canon = constrain_points_to_ray(
                        keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
                    )
                    keyframe.X_canon = X_canon.squeeze(0)
                pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
                color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
                score = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)

                valid = (
                    keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
                    > self.c_confidence_threshold
                )
                pointclouds.append(pW[valid])
                colors.append(color[valid])
                confidence_scores.append(score[valid])

                if self.segment_point_cloud:
                    seg_mask = keyframe.img_segmentation_mask.cpu().numpy().astype(np.uint8).reshape(-1)
                    segmentation_masks.append(seg_mask[valid])

            pointclouds = np.concatenate(pointclouds, axis=0)
            colors = np.concatenate(colors, axis=0)
            confidence_scores = np.concatenate(confidence_scores, axis=0)
            if self.segment_point_cloud:
                segmentation_masks = np.concatenate(segmentation_masks,axis=0)

        #update the pointcloud with every frame, but only have parts of the color
        elif self.point_cloud_method == "refreshing" and add_new_kf:

            if self.segment_point_cloud:
                raise NotImplementedError("segmentation currently only available for accumulating mode.")

            if config["use_calib"] and calibration_K is not None:
                X_canon = constrain_points_to_ray(
                    frame.img_shape.flatten()[:2], frame.X_canon[None], calibration_K
                )
                frame.X_canon = X_canon.squeeze(0)
                
            pW = T_WC.act(frame.X_canon).cpu().numpy().reshape(-1, 3)
            color = (frame.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
            score = frame.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            valid = (
                    frame.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
                    > self.c_confidence_threshold
                )

            pointclouds = pW
            colors = color
            confidence_scores = score

        else: 
            raise ValueError("Invalid variant")
                



        point_cloud_data_entity = PointCloudDataEntity(
            point_cloud=pointclouds, #pointcloud in world coordinates
            rgb= colors,
            confidence_scores=confidence_scores,
            segmentation_mask = segmentation_masks
        )

        return {
            "point_cloud": point_cloud_data_entity,
            "camera_pose": T_WC,
            "key_frame_flag": add_new_kf
        }