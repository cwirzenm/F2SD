from dataset import CustomDataset
import shutil
import os


class DatasetGenerator:
    def __init__(self, root: str, **kwargs):
        self.root = root
        self.cwd = os.getcwd()
        self.kwargs = kwargs

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
    with DatasetGenerator("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\testing_consistory\\1") as tup:
        a, b = tup
        print(str(a))
        print(str(b))
