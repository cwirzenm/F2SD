from dataset import CustomDataset
import shutil
import glob
import os


class DatasetGenerator:
    def __init__(self, root: str, seq_length=4, **kwargs):
        self.root = root
        self.cwd = os.getcwd()
        self.kwargs = kwargs
        self.source, *self.targets = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        self.chunks = [self.targets[i:i + seq_length] for i in range(0, len(self.targets), seq_length)]
        self.seq_length = seq_length
        self.counter = -1

    def __iter__(self):
        os.chdir(self.root)
        os.makedirs('source', exist_ok=True)
        os.makedirs('target', exist_ok=True)
        for i in range(self.seq_length):
            shutil.copy(self.source, f"source/{i}_{self.source}")
        self.source_path = os.path.join(self.root, 'source')
        return self

    def __next__(self):
        self.counter += 1
        if self.counter >= len(self.chunks):
            shutil.rmtree('source/')
            shutil.rmtree('target/')
            os.chdir(self.cwd)
            raise StopIteration

        old_files = glob.glob('target/*')
        for f in old_files: os.remove(f)
        for t in self.chunks[self.counter]:
            shutil.copy2(t, 'target/')

        target_path = os.path.join(self.root, 'target')
        return CustomDataset(self.source_path, **self.kwargs), CustomDataset(target_path, **self.kwargs)

    def __enter__(self):
        os.chdir(self.root)
        os.makedirs('target', exist_ok=True)
        os.makedirs('source', exist_ok=True)
        source, *self.targets = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        for i, t in enumerate(self.targets):
            shutil.copy(source, f"source/{i}_{source}")
            shutil.copy2(t, 'target/')
        self.source = os.path.join(self.root, 'source')
        self.target = os.path.join(self.root, 'target')
        return CustomDataset(self.source, **self.kwargs), CustomDataset(self.target, **self.kwargs)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        shutil.rmtree('source/')
        shutil.rmtree('target/')
        os.chdir(self.cwd)


if __name__ == '__main__':
    d = DatasetGenerator("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\temporalstory\\flintstones\\wilma")
    for x, y in d:
        print(x, y)
    print('done')
