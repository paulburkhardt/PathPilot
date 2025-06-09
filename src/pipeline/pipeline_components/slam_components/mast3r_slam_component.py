# Add the MASt3r-SLAM root directory to sys.path if not already present
import sys
import os
mast3r_slam_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mast3r_slam"))
if mast3r_slam_root not in sys.path:
    sys.path.insert(0, mast3r_slam_root)

import time
from typing import List, Dict, Any, Tuple
from .abstract_slam_component import AbstractSLAMComponent
import torch.multiprocessing as mp
import torch
import lietorch

from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config




def relocalization(frame, keyframes, factor_graph, retrieval_database):
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
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

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
        img_height (int): The height of the input images.
        img_width (int): The width of the input images.
        device (str, optional): The device on which to run MAST3R SLAM. Defaults to "cuda:0".
    """
    
    def __init__(
        self,
        mast3r_slam_config_path: str, 
        device: str = "cuda:0",
        )->None:

        super().__init__()

        self.mast3r_config = load_config(mast3r_slam_config_path)
        
        #prepare MAST3R Slam init based on their main
        mp.set_start_method("spawn")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)
    
        self.manager = mp.Manager()
        self.device = device


        self._is_inited = False

        

    @property
    def inputs_from_bucket(self) -> List[str]:
        """This component requires RGB images as input."""
        return ["rgb_image","step_nr","timestamp","img_size","img_width","img_height"]
    
    @property
    def outputs_to_bucket(self) -> List[str]:
        """This component outputs point clouds and camera poses."""
        return ["point_cloud", "camera_pose"]
    

    def _init_rest(
            self,
            img_height:int,
            img_width: int):
        """
        inits the stuff that is dataset parameter dependent
        """

        self.keyframes = SharedKeyframes(self.manager,img_height, img_width)
        self.states = SharedStates(self.manager, img_height, img_width)

        self.model = load_mast3r(device=self.device,path="MASt3R-SLAM/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" )
        self.model.share_memory()


        #TODO: Add option to use calibation. Maybe add calibration to data bucket
        self.K = None

        self.tracker = FrameTracker(self.model,self.keyframes,self.device)
        self.backend = mp.Process(target=run_backend, args=(self.mast3r_config, self.model, self.states, self.keyframes, self.K))
        self.backend.start()


    def __del__(self):
        """
        Clean up on delete

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
             rgb_image: Any,
             timestamp: Any, 
             step_nr: int,
             img_size:int,
             img_width: int,
             img_height: int) -> Dict[str, Any]:
        """
        Process an RGB image using MAST3R SLAM.
        
        Args:
            rgb_image: The input RGB image
            timestamp: Timestamp of the image within the video
            step_nr: Frame number
            
        Raises:
            NotImplementedError: As this is currently a placeholder
        """
        

        if not self._is_inited:
            self._init_rest(img_height=img_height,img_width=img_width)
            self._is_inited = True

        mode = self.states.get_mode()
        
        T_WC = (
            lietorch.Sim3.Identity(1, device=self.device)
            if step_nr == 0
            else self.states.get_frame().T_WC
        )
        frame = create_frame(step_nr, rgb_image, T_WC, img_size=img_size, device=self.device)



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


        if X_init is not None and C_init is not None:
            X = X_init
            C = C_init
            
        return {
            "point_cloud": X,
            "camera_pose": T_WC
        }