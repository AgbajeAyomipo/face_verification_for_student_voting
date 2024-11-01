import pandas as pd
import os
import tqdm
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib import image


def make_records():
    mat_no = list()
    for i in tqdm(os.listdir("records")):
        mat_no.append("/".join(i.split("-")))


    img_path = list()
    for i in tqdm(os.listdir("records")):
        record_ = f"records/{i}"
        rec_ = list()
        for n, j in enumerate((os.listdir(record_))):
            rec_.append(f"records/{i}/{j}")
        img_path.append(" ".join(rec_)) 

    records_df = pd.DataFrame(
    data = {
        "matric number": mat_no,
        "img paths": img_path
    }
    )
    records_df.to_csv("records.csv", index=0)
    return records_df

def empty_img():
    img__ = image.imread("no_img.jpg")
    return img__



# if __name__ == "__main__":
#     make_records()