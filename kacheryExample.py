import kachery as ka

path = 'sha1:////3476867b4d9300e4a44e2b910af87b08f8e608bf/dataset.csv'

# Get the path to the cached file
path_local = ka.load_file(path)

# Or load the text of the file directly
dataset_csv_text = ka.load_text(path)
