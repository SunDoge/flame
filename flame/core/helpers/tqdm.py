from tqdm import tqdm

def tqdm_get_rate(pbar: tqdm) -> float:
    return pbar.format_dict['rate']



 