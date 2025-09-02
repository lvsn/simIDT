cd Render

export objects_path=../Data/demo_pem/objects_behave

for filename in "$objects_path"/*; do
    echo "put ${filename}"
    blenderproc run render_custom_templates.py --cad_path $filename #--colorize True 
done