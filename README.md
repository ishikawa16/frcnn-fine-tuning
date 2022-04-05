# frcnn-fine-tuning
This is a repository for fine-tuning object detectors using a Pytorch implementation of Faster R-CNN.

## Requirement
- Python 3.8.10

## Usage
1. Clone repository.
    ```
    git clone https://github.com/ishikawa16/frcnn-fine-tuning.git
    cd frcnn-fine-tuning
    ```

1. Create python environment.
    ```
    pyenv virtualenv 3.8.10 fft
    pyenv local fft
    pip install -r requirements.txt
    ```

1. Modify `dataset_dir` in `config.json` that specifies the location of your dataset.  
    e.g.
    ```
    dataset_dir = "data/alfred_dataset/"
    ```

    Dataset should have the following directory structure:
    ```
    DATASET
    ├── train.jsonl
    ├── valid.jsonl
    └── image
        ├── 000001.jpg
        ├── 000002.jpg
        ├── 000003.jpg
        └── ...
    ```
    In `train.jsonl` and `valid.jsonl`, each record should contain the keys: "id" and "objects."  
    e.g.
    ```
    {
        "id": 000001,
        "objects":[
            {
                "label": apple,
                "box": [10, 20, 40, 50]
            },
            {
                "label": orange,
                "box": [50, 80, 90, 110]
            },
            ...
            {
                "label": lemon,
                "box": [30, 15, 65, 45]
            }
        ]
    }
    ```
    "box" denotes its position \[left, upper, right, lower\].

1. Modify `classes` in `config.json` that covers the full range of labels.

    Note that `classes` must include the class "\_\_background\_\_".  
    e.g.
    ```
    classes = [
        "__background__",
        "apple",
        "orange",
        "lemon"
    ]
    ```

1. Run fine-tuning.
    ```
    python src/main.py \
        --mode train \
        --config config.json
    ```
    The models will be saved at `model/model_exx.pth`.

1. Run inference with an image.
    ```
    python src/main.py \
        --mode predict_oneshot \
        --config config.json \
        --image IMAGE
    ```
    The result file will be written at `output/result.json`.  
    If you would like to extract the feature of each bounding box, enable `--save_features`.
