import gc

import cv2
import numpy as np
import torch


class _SAM3TubeletPredictor:
    """SAM2-style adapter around SAM3's video model.

    TubeletGraph's tubelet code expects a predictor with init_state,
    reset_state, add_new_mask, and propagate_in_video. SAM3 exposes the
    same mask-tracker internally, but its public video predictor returns a
    different dictionary format, so this class normalizes the interface.
    """

    def __init__(
        self,
        checkpoint_path=None,
        bpe_path=None,
        device="cuda",
        strict_state_dict_loading=True,
        apply_temporal_disambiguation=True,
        compile=False,
        **kwargs,
    ):
        try:
            from sam3.model_builder import build_sam3_video_model
        except ImportError as exc:
            raise ImportError(
                "SAM3 is not installed. Clone facebookresearch/sam3 into "
                "thirdparty/sam3, run `pip install -e thirdparty/sam3`, "
                "accept access to https://huggingface.co/facebook/sam3, "
                "and run `hf auth login`."
            ) from exc

        self.device = device
        self.model = build_sam3_video_model(
            checkpoint_path=checkpoint_path,
            bpe_path=bpe_path,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            device=device,
            compile=compile,
            **kwargs,
        )
        self.model.eval()

    def init_state(self, video_path=None, resource_path=None, **kwargs):
        resource_path = resource_path or video_path
        if resource_path is None:
            raise ValueError("SAM3 init_state requires `video_path` or `resource_path`.")

        state = self.model.init_state(resource_path=resource_path, **kwargs)
        self._init_tubelet_state(state)
        return state

    def reset_state(self, inference_state):
        self.model.reset_state(inference_state)
        self._init_tubelet_state(inference_state)

    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        add_mask_to_memory=True,
        **kwargs,
    ):
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.model.device)
        if mask_t.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {tuple(mask_t.shape)}.")

        tracker_states = self.model._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            new_obj_ids=[int(obj_id)],
            new_obj_masks=mask_t.unsqueeze(0),
            tracker_states_local=inference_state["tracker_inference_states"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            feature_cache=inference_state["feature_cache"],
        )
        inference_state["tracker_inference_states"] = tracker_states
        inference_state["previous_stages_out"][frame_idx] = "_TUBELET_MASK_PROMPT_"
        inference_state["obj_idx_to_id"][int(obj_id)] = int(obj_id)
        inference_state["output_dict_per_obj"].setdefault(
            int(obj_id), {"non_cond_frame_outputs": {}}
        )
        return frame_idx, [int(obj_id)], mask_t.unsqueeze(0).unsqueeze(0)

    def propagate_in_video(self, inference_state, **kwargs):
        for frame_idx, outputs in self.model.propagate_in_video(
            inference_state=inference_state, **kwargs
        ):
            obj_ids, masks, scores = self._normalize_outputs(outputs, inference_state)
            self._store_scores(inference_state, frame_idx, obj_ids, scores)
            yield frame_idx, obj_ids, masks

    def _init_tubelet_state(self, state):
        state["obj_idx_to_id"] = {}
        state["output_dict_per_obj"] = {}

    def _normalize_outputs(self, outputs, inference_state):
        height = inference_state["orig_height"]
        width = inference_state["orig_width"]
        if outputs is None:
            return [], torch.zeros(0, 1, height, width, device=self.model.device), []

        obj_ids = [int(x) for x in outputs.get("out_obj_ids", [])]
        masks = outputs.get("out_binary_masks", None)
        scores = outputs.get("out_probs", np.ones(len(obj_ids), dtype=np.float32))

        if masks is None or len(obj_ids) == 0:
            mask_t = torch.zeros(0, 1, height, width, device=self.model.device)
        else:
            mask_t = torch.as_tensor(masks, dtype=torch.float32, device=self.model.device)
            if mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(1)
            mask_t = (mask_t > 0).float()

        return obj_ids, mask_t, [float(x) for x in scores]

    def _store_scores(self, inference_state, frame_idx, obj_ids, scores):
        for obj_id, score in zip(obj_ids, scores):
            obj_out = inference_state["output_dict_per_obj"].setdefault(
                int(obj_id), {"non_cond_frame_outputs": {}}
            )
            obj_out["non_cond_frame_outputs"][int(frame_idx)] = {
                "object_score_logits": torch.tensor([[score]], device=self.model.device)
            }


class SAM3:
    def __init__(
        self,
        model_weights=None,
        bpe_path=None,
        device="cuda",
        multi_mask=False,
        multi_mask_fallback="morphology",
        strict_state_dict_loading=True,
        apply_temporal_disambiguation=True,
        compile=False,
        **kwargs,
    ):
        self.predictor = _SAM3TubeletPredictor(
            checkpoint_path=model_weights,
            bpe_path=bpe_path,
            device=device,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            compile=compile,
            **kwargs,
        )
        self.multi_mask = multi_mask
        self.multi_mask_fallback = multi_mask_fallback
        self.inference_state = None

    def initialize(self, **info):
        self.inference_state = self.predictor.init_state(video_path=info["video_dir"])

    def track(self, mask, frame_idx=0, clean_prev_memory=20):
        self.predictor.reset_state(self.inference_state)
        self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=mask,
        )

        output = {"prediction": {}, "obj_score": {}}
        if self.multi_mask:
            output["multi_masks"] = {}
            output["multi_masks_pred_ious"] = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state
        ):
            output["prediction"][out_frame_idx] = {
                obj_id: (out_mask_logits[i, 0] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(out_obj_ids)
            }
            output["obj_score"][out_frame_idx] = self._scores_for_frame(
                out_frame_idx, out_obj_ids
            )

            if self.multi_mask:
                output["multi_masks"][out_frame_idx] = {}
                output["multi_masks_pred_ious"][out_frame_idx] = {}
                for obj_id, obj_mask in output["prediction"][out_frame_idx].items():
                    masks = self._make_multi_mask_fallback(obj_mask)
                    output["multi_masks"][out_frame_idx][obj_id] = masks
                    output["multi_masks_pred_ious"][out_frame_idx][obj_id] = {
                        rank: 1.0 if rank == 0 else 0.0 for rank in masks
                    }

            self._drop_old_scores(out_frame_idx - clean_prev_memory)
            torch.cuda.empty_cache()

        return output

    def clear_all_cache(self):
        if self.inference_state is not None:
            for key in list(self.inference_state.keys()):
                self.inference_state[key] = None
            self.inference_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _scores_for_frame(self, frame_idx, obj_ids):
        scores = {}
        for obj_id in obj_ids:
            obj_out = self.inference_state["output_dict_per_obj"].get(obj_id, {})
            frame_out = obj_out.get("non_cond_frame_outputs", {}).get(frame_idx)
            if frame_out is not None:
                scores[obj_id] = frame_out["object_score_logits"][0, 0].item()
        return scores

    def _drop_old_scores(self, frame_idx):
        if frame_idx < 0:
            return
        for obj_output in self.inference_state["output_dict_per_obj"].values():
            obj_output["non_cond_frame_outputs"].pop(frame_idx, None)

    def _make_multi_mask_fallback(self, mask):
        mask = np.asarray(mask).astype(bool)
        if self.multi_mask_fallback != "morphology":
            return {0: mask, 1: mask.copy(), 2: mask.copy()}

        mask_u8 = mask.astype(np.uint8)
        kernel = np.ones((5, 5), dtype=np.uint8)
        eroded = cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
        dilated = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)
        return {0: mask, 1: eroded, 2: dilated}
