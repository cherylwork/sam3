import numpy as np
import torch


class SAM3:
    def __init__(self,
                 model_weights=None,
                 bpe_path=None,
                 device="cuda",
                 multi_mask=False,
                 strict_state_dict_loading=True,
                 apply_temporal_disambiguation=True,
                 compile=False,
                 **kwargs):

        from sam3.model_builder import build_sam3_video_model
        self.model = build_sam3_video_model(
            checkpoint_path=model_weights,
            bpe_path=bpe_path,
            strict_state_dict_loading=strict_state_dict_loading,
            apply_temporal_disambiguation=apply_temporal_disambiguation,
            device=device,
            compile=compile,
            **kwargs,
        )
        self.model.eval()

        self.predictor = self
        self.multi_mask = multi_mask
        self.inference_state = None

    def initialize(self, **info):
        self.inference_state = self.predictor.init_state(video_path=info['video_dir'])

    def track(self, mask, frame_idx=0, clean_prev_memory=20):
        self.predictor.reset_state(self.inference_state)
        _, _, _ = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=mask,
        )

        # output
        # |-> "prediction"
        #     |-> frame_idx
        #         |-> obj_id: np.ndarray
        # |-> "obj_score"
        #     |-> frame_idx
        #         |-> obj_id: float

        output = {'prediction': dict(), 'obj_score': dict()}

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            output['prediction'][out_frame_idx] = {
                obj_idx: (out_mask_logits[i, 0] > 0.0).cpu().numpy()
                    for i, obj_idx in enumerate(out_obj_ids)
            }

            for obj_idx in out_obj_ids:
                obj_out = self.inference_state['output_dict_per_obj'][obj_idx]
                if out_frame_idx not in obj_out['non_cond_frame_outputs']:
                    continue
                obj_frame_out = obj_out['non_cond_frame_outputs'][out_frame_idx]

                if out_frame_idx not in output['obj_score']:
                    output['obj_score'][out_frame_idx] = dict()
                output['obj_score'][out_frame_idx][obj_idx] = obj_frame_out['object_score_logits'][0,0].item()

            delete_f_idx = out_frame_idx - clean_prev_memory
            for obj_idx, obj_output in self.inference_state['output_dict_per_obj'].items():
                if delete_f_idx in obj_output['non_cond_frame_outputs'].keys():
                    obj_output['non_cond_frame_outputs'][delete_f_idx].clear()
            torch.cuda.empty_cache()

        return output

    def clear_all_cache(self):
        if self.inference_state is not None:
            for k in self.inference_state.keys():
                self.inference_state[k] = None
        torch.cuda.empty_cache()

    def init_state(self, video_path=None, resource_path=None, **kwargs):
        resource_path = resource_path or video_path
        if resource_path is None:
            raise ValueError("SAM3 init_state requires `video_path` or `resource_path`.")

        inference_state = self.model.init_state(resource_path=resource_path, **kwargs)
        self._init_tubelet_state(inference_state)
        return inference_state

    def reset_state(self, inference_state):
        self.model.reset_state(inference_state)
        self._init_tubelet_state(inference_state)

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask, **kwargs):
        mask = torch.as_tensor(mask, dtype=torch.float32, device=self.model.device)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {tuple(mask.shape)}.")

        self._cache_frame_features(inference_state, frame_idx)

        inference_state['tracker_inference_states'] = self.model._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=inference_state['num_frames'],
            new_obj_ids=[int(obj_id)],
            new_obj_masks=mask.unsqueeze(0),
            tracker_states_local=inference_state['tracker_inference_states'],
            orig_vid_height=inference_state['orig_height'],
            orig_vid_width=inference_state['orig_width'],
            feature_cache=inference_state['feature_cache'],
        )
        inference_state['previous_stages_out'][frame_idx] = '_TUBELET_MASK_PROMPT_'
        inference_state['obj_idx_to_id'][int(obj_id)] = int(obj_id)
        inference_state['output_dict_per_obj'].setdefault(
            int(obj_id), {'non_cond_frame_outputs': dict()}
        )

        return frame_idx, [int(obj_id)], mask.unsqueeze(0).unsqueeze(0)

    def propagate_in_video(self, inference_state, **kwargs):
        frame_outputs = dict()
        for tracker_state in inference_state['tracker_inference_states']:
            for frame_idx, obj_ids, _, video_res_masks, obj_scores in self.model.tracker.propagate_in_video(
                tracker_state,
                start_frame_idx=kwargs.get('start_frame_idx', None),
                max_frame_num_to_track=kwargs.get('max_frame_num_to_track', None),
                reverse=kwargs.get('reverse', False),
                tqdm_disable=True,
            ):
                if frame_idx not in frame_outputs:
                    frame_outputs[frame_idx] = {'obj_ids': [], 'masks': [], 'scores': []}
                frame_outputs[frame_idx]['obj_ids'] += [int(obj_id) for obj_id in obj_ids]
                frame_outputs[frame_idx]['masks'].append(video_res_masks)
                frame_outputs[frame_idx]['scores'].append(obj_scores.reshape(-1))

        for frame_idx in sorted(frame_outputs.keys()):
            out_obj_ids = frame_outputs[frame_idx]['obj_ids']
            out_mask_logits = torch.cat(frame_outputs[frame_idx]['masks'], dim=0)
            out_scores = torch.cat(frame_outputs[frame_idx]['scores'], dim=0).tolist()
            self._store_output_scores(inference_state, frame_idx, out_obj_ids, out_scores)
            yield frame_idx, out_obj_ids, out_mask_logits

    def _init_tubelet_state(self, inference_state):
        inference_state['obj_idx_to_id'] = dict()
        inference_state['output_dict_per_obj'] = dict()

    def _cache_frame_features(self, inference_state, frame_idx):
        if frame_idx in inference_state['feature_cache']:
            return

        self.model.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=inference_state['num_frames'],
            reverse=False,
            input_batch=inference_state['input_batch'],
            geometric_prompt=inference_state['constants']['empty_geometric_prompt'],
            feature_cache=inference_state['feature_cache'],
            allow_new_detections=False,
        )

    def _format_output(self, output, inference_state):
        height = inference_state['orig_height']
        width = inference_state['orig_width']

        if output is None:
            return [], torch.zeros(0, 1, height, width, device=self.model.device), []

        out_obj_ids = [int(x) for x in output.get('out_obj_ids', [])]
        out_scores = output.get('out_probs', np.ones(len(out_obj_ids), dtype=np.float32))
        out_masks = output.get('out_binary_masks', None)

        if out_masks is None or len(out_obj_ids) == 0:
            out_mask_logits = torch.zeros(0, 1, height, width, device=self.model.device)
        else:
            out_mask_logits = torch.as_tensor(out_masks, dtype=torch.float32, device=self.model.device)
            if out_mask_logits.ndim == 3:
                out_mask_logits = out_mask_logits.unsqueeze(1)
            out_mask_logits = (out_mask_logits > 0).float()

        return out_obj_ids, out_mask_logits, [float(score) for score in out_scores]

    def _store_output_scores(self, inference_state, frame_idx, out_obj_ids, out_scores):
        for obj_idx, score in zip(out_obj_ids, out_scores):
            obj_out = inference_state['output_dict_per_obj'].setdefault(
                int(obj_idx), {'non_cond_frame_outputs': dict()}
            )
            obj_out['non_cond_frame_outputs'][int(frame_idx)] = {
                'object_score_logits': torch.tensor([[score]], device=self.model.device)
            }
