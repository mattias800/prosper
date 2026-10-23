# Renderer resource-contract implementations shared by every existing consumer.
# Keep the target inventory aligned with the render_runner.h dependency audit.
if(TARGET Vulkan::Vulkan)
  set(_prosper_render_resource_consumers
    boot_trace
    gpu_replay
    prosper_live_renderer
    test_backend_persistent_resource_lock
    test_backend_timestamp_pool
    test_broadcast_htile_clear
    test_cached_compute_fill
    test_color_target_create_failure
    test_command_pool_reuse
    test_compute_linear_row_pitch
    test_compute_overwrite_eviction
    test_compute_quad_or_exec
    test_compute_scanout
    test_dcc_clear_render_target
    test_depth_array_source_snapshot
    test_depth_bounds_render
    test_depth_cube_gpu
    test_depth_cube_source_snapshot
    test_depth_only_attachment_extent
    test_descriptor_array_render
    test_draw_resource_status
    test_ds_guest_plane_bytes
    test_ds_layer_stride
    test_ds_replay_alias
    test_dsbridge_miss_classifier
    test_float32_array_gather
    test_float_transport_render
    test_fragment_draw_exec
    test_fragment_loop_mask_exec
    test_fragment_loop_vcc_scratch
    test_fragment_numeric_mbcnt
    test_fragment_partial_wave_exec
    test_fragment_proven_votes
    test_fragment_scalar_bank
    test_fragment_scalar_bank_gpu
    test_fragment_scalar_pair_mask
    test_fragment_vote_refusal_reporting
    test_game_compute_rtt_mirror
    test_gds_scan_memo
    test_gpu_capture_render
    test_gpu_detile_render
    test_gpu_detile_storage
    test_gpu_execute
    test_gpustate_render
    test_graphics_cache_persistence
    test_graphics_raw_wide_upload
    test_guest_write_drain
    test_host_read_barrier
    test_indexed_render
    test_integer_color_export
    test_interp_render
    test_live_compute_bgra_target
    test_live_compute_depth_unorm16
    test_live_compute_narrow_float16
    test_live_renderer_registration_statics
    test_live_target_format
    test_mapped_staging
    test_mesh_backend
    test_mimg_nsa_forms_render
    test_mip_assembly_barrier
    test_multidraw_render
    test_native_bc_upload
    test_native_mask_participation_exec
    test_ngg_subgroup_backend
    test_ordered_graphics_read_point
    test_owned_graphics_wave_live
    test_persistent_readback_refusal
    test_pipeline_render
    test_present_blit
    test_ps_pull_model_exec
    test_raster_quad_collection
    test_real_shader_render
    test_recompiled_fragment
    test_recompiled_shaders
    test_rect_synthesis_render
    test_render_buffer_residency
    test_render_diagnostic_paths
    test_render_object_create_failure
    test_renderer_uint_copy
    test_scc_do_while_loops
    test_shadow_compare_render
    test_shared_vulkan_device
    test_storage_metadata_borrow
    test_storage_rtt_gpu_seed
    test_submit_gate_wiring
    test_texture_mip_render
    test_texture_sample_render
    test_textured_interp_render
    test_vertex_fetch_render
    test_volume_multi_target
    test_volume_target_handoff
    test_wave64_host_announcement
  )
  set(_prosper_render_resources_needed OFF)
  foreach(_consumer IN LISTS _prosper_render_resource_consumers)
    if(TARGET ${_consumer})
      set(_prosper_render_resources_needed ON)
    endif()
  endforeach()
  if(_prosper_render_resources_needed)
    add_library(prosper_render_resources STATIC tests/fixtures/render_resource_contracts.cpp)
    target_include_directories(prosper_render_resources PRIVATE src tests frontends)
    target_link_libraries(prosper_render_resources PRIVATE Vulkan::Vulkan)
    foreach(_consumer IN LISTS _prosper_render_resource_consumers)
      if(TARGET ${_consumer})
        target_link_libraries(${_consumer} PRIVATE prosper_render_resources)
      endif()
    endforeach()
  endif()
  unset(_prosper_render_resources_needed)
  unset(_prosper_render_resource_consumers)
  unset(_consumer)
endif()
