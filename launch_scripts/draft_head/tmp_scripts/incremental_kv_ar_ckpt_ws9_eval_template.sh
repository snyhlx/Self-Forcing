cd /root/Self-Forcing

CUDA_VISIBLE_DEVICES=0 \
WAN_FLASH_ATTN_VERSION=2 \
WAN_ATTENTION_BACKEND=flash \
CAUSAL_WAN_FLEX_COMPILE_MODE=none \
/root/Self-Forcing/sf_venv/bin/python sdvg_inference.py heuristic \
  --config_path /root/Self-Forcing/configs/self_forcing_dmd.yaml \
  --model_root /mnt/lanxiangh/models \
  --draft_model_name Wan2.1-T2V-1.3B \
  --draft_checkpoint_path /mnt/lanxiangh/models/Self-Forcing/checkpoints/self_forcing_dmd.pt \
  --target_model_name Wan2.1-T2V-14B \
  --target_checkpoint_path /mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors \
  --draft_head_checkpoint_path /mnt/lanxiangh/checkpoints/specgen/20260623_103259_causal_wan_ar_prefixtokens_krea14b_unipc5_shift8_flow_ep3_lr1e-6_bf16_fulldata_dh_causal_wan_ar_flow_teacher_trajectory_h5120_ctx4680_p1x1x1_st999_969_922_841_666_bs1_g8_lr1e-6_fl1_0_ampbf16_gc1_fw0_cfg495f0fd7__allspan_ep3_bf16/final.pt \
  --mode draft_head \
  --draft_head_inference_mode incremental_kv \
  --output_decode_mode chunk_streaming \
  --also_save_target_video \
  --agreement_metric rmse \
  --denoising_step_list "999 969 922 841 666" \
  --num_blocks 9 \
  --seed 42 \
  --output_dir /root/Self-Forcing/outputs/draft_head/test_runs/eval_20260623_103259_incremental_kv_allspan_draft_head_chunk_streaming_with_target_train_0_5 \
  --prompt_file /root/Self-Forcing/prompts/MovieGenVideoBench.txt \
  --start_index 0 \
  --max_prompts 5
