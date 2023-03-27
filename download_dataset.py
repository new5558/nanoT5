import datasets

dataset = datasets.load_dataset(
    'mc4',
    'th',
    # streaming = True
    download_config = datasets.utils.DownloadConfig(num_proc = 8)
)

print(dataset)