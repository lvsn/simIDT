SEQUENCES=(
    "Date01_Sub01_boxlarge_hand"
    "Date01_Sub01_chairblack_lift"
    "Date01_Sub01_chairblack_hand"
)

for SEQ_NAME in "${SEQUENCES[@]}"; do
    echo "Processing sequence: $SEQ_NAME"

    export SEQ_PATH=/home-local2/chren50.extra.nobkp/behave/test/frames_resized/$SEQ_NAME 

    mkdir Data/demo_pem/output_test_compile/$SEQ_NAME
    # Run the Python script with the current object
    python evaluate_sequence_tracker_behave_mt.py \
        --data_dir Data/demo_pem \
        --output_dir Data/demo_pem/output_test_compile/$SEQ_NAME \
        --seq_path $SEQ_PATH \
        --fp_init \
        > Data/demo_pem/output_test_compile/$SEQ_NAME/seq_output.txt

    echo "Finished processing object"
    echo "----------------------------------"
done

# cd eval
# python compile_results_memoire.py \
#     --dir ../Data/demo_pem/output_mem/behave \
#     --data_dir ../Data/demo_pem \
#     --seq_dir /home-local2/chren50.extra.nobkp/behave/test/frames_resized