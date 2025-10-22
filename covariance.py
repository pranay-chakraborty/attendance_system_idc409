import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _(X):
    import numpy as np
    import os
    from PIL import Image




    def load_data_frm_subfolders(
        base_dir):
        image_vectors = []
        labels = []
        image_shape = None
        print(f'scanning base directory: {base_dir}')

        #find all subdir, each one being a subject
        subject_dirs = sorted([d for d in os.listdir(base_dir)
                              if os.path.isdir(os.path.join(base_dir, d))])
        if not subject_dirs:
            raise FileNotFoundError(f"No subject subdir found in base_dir")
    
        # Assign label to each sub
        label_numeric = 0
        for subject_name in subject_dirs:
            subject_path = os.path.join(base_dir, subject_name)
            print(f"loading image for sub: {subject_name} of label {label_numeric}")
            image_files = sorted([f for f in os.listdir(subject_path)
                                 if f.lower(). endswith('.pgm')])
            if not image_files:
                print(f'Warning: No .pgm files found')
                continue
            for filename in image_files:
                try:
                    img_path = os.path.join(subject_path, filename)
                    img = Image.open(img_path).convert('L')
                    img_data = np.array(img)
                    if image_shape is None:
                        image_shape = img_data,image_shape
                        print(f'Face img obtained, dimensions : {image_shape}')
                    if image_shape != image_shape:
                        print(f'Skipping {filename}: shape mismatch')
                    continue 

                    image_vectors.append(img_data.flatten())
                    labels.append(label_numeric)
                except Exception as e:
                    print(f'error loading {filename}: {e}')
            label_numeric +=1

        if not image_vectors:
            raise ValueError('No images were loaded. stopping')

        x = np.vstack(image_vectors)
        y = np.array(labels)

        return X, y, image_shape

    load_data_frm_subfolders(base_dir='face-dataset')
    return


if __name__ == "__main__":
    app.run()
