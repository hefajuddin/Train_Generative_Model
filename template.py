import os, sys
from pathlib import Path
import logging

listOfFiles=[
    "pretrained_model.py",
    "model_tokenize.py",
    "trainer.py",
    f"templates/index.html",
    f"data/context.py",
    f"data/dataset.py",
    f"data/collator.py",
    ".env",
    "config.py",
    "requirements.txt",
    "app.py",
    "upload.py"
]

for path in listOfFiles:
    filepath=Path(path)
    filedir, filename=os.path.split(path)

    if filedir!="":
        os.makedirs(filedir, exist_ok=True)

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "w") as f:
            pass

    else:
        logging.info("file is already present at :{filepath}" )

