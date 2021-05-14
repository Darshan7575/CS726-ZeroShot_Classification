# CS726-ZeroShot_Classification
Codebase for the Zeroshot Classification project as a part of CS726 curriculum

## Install the following pre requisites needed for running the model
1. `pip install -U sentence-transformers`
2. `pip install pydantic, fastapi, colabcode, python-multipart, lime, aiofiles`

## To run the project execute the following steps in order
1. Download the pretrained weights from here: [drive link](https://drive.google.com/drive/folders/1ppuBPhij6JzBJocIw7WQCNHclSq7bvB3?usp=sharing)
2. Place the `weights.pt` file downloaded from the above link in the empty directory named `model`
3. Run the backend using the command `python3 model_backend.py` 
4. Go to UI directory and update the variable `site_url` in `way01/views.py` file with the link generated from step 3
5. Execute `python3 manage.py runserver` command from UI directory
6. You are good to go. Visit `localhost:8000` in your browser and you should be able to see the ui.
