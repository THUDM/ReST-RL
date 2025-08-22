from datasets import Dataset


def split_dataset(dataset: Dataset):
    train_ratio = 0.95
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    if test_size < 100:
        print('Test size is too small, set to 100 by default\n')
        test_size = 100
        train_size = len(dataset) - test_size
        assert train_size > test_size, "Dataset is too small, please enlarge the dataset to ensure valid training results\n"
    dataset_split = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    return dataset_split['train'], dataset_split['test']
