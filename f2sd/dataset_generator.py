from dataset import CustomDataset
import shutil
import glob
import os


class DatasetGenerator:
    def __init__(self, root: str, seq_length=4, **kwargs):
        self.root: str = root
        self.cwd: str = os.getcwd()
        self.kwargs: dict = kwargs
        self.source, *self.targets = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]
        self.chunks: list = [self.targets[i:i + seq_length] for i in range(0, len(self.targets), seq_length)]
        self.seq_length: int = seq_length
        self.counter: int = -1

    def __iter__(self):
        os.chdir(self.root)
        os.makedirs('source', exist_ok=True)
        os.makedirs('target', exist_ok=True)

        old_files: list = glob.glob('source/*')
        for f in old_files: os.remove(f)
        for i in range(self.seq_length):
            shutil.copy(self.source, f"source/{i}_{self.source}")
        self.source_path: str = os.path.join(self.root, 'source')
        return self

    def __next__(self) -> tuple[CustomDataset, CustomDataset]:
        self.counter += 1
        if self.counter >= len(self.chunks):
            shutil.rmtree('source/')
            shutil.rmtree('target/')
            os.chdir(self.cwd)
            raise StopIteration

        old_targets: list = glob.glob('target/*')
        for f in old_targets: os.remove(f)

        # final chunk may be smaller than sequence length therefore we need to make sure the source folder size is the same
        old_source: list = glob.glob('source/*')
        curr_chunk_size = len(self.chunks[self.counter])
        temp_counter = 0
        for f in old_source:
            if (len(old_source) - temp_counter) == curr_chunk_size:
                break
            os.remove(f)
            temp_counter += 1

        for t in self.chunks[self.counter]:
            shutil.copy2(t, 'target/')

        target_path: str = os.path.join(self.root, 'target')
        return CustomDataset(self.source_path, **self.kwargs), CustomDataset(target_path, **self.kwargs)


if __name__ == '__main__':
    d = DatasetGenerator("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\temporalstory\\flintstones\\wilma")
    for x, y in d:
        print(x, y)
    print('done')
