export CAD_PATH=../Data/demo_multi/objects/clock    # path to a given cad model(mm) folder
export RGB_PATH=../Data/demo_multi/rgb/00054_clock.png           # path to a given RGB image
export DEPTH_PATH=../Data/demo_multi/depth/00054_clock.png       # path to a given depth map(mm)
export CAMERA_PATH=../Data/demo_multi/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=../Data/out_debug         # path to a pre-defined file for saving results


# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 


# Run instance segmentation model
export SEGMENTOR_MODEL=sam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --stability_score_thresh 0.5


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

