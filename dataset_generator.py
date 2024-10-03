from dataset import CustomDataset
import shutil
import os


class DatasetGenerator:
    def __init__(self, root: str):
        os.chdir(root)
        os.makedirs('target', exist_ok=True)
        os.makedirs('source', exist_ok=True)
        source, *self.targets = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        shutil.copy2(source, 'source/')
        self.source = os.path.join(root, 'source')
        self.target = os.path.join(root, 'target')

    def __iter__(self):
        for target in self.targets:
            shutil.copy2(target, 'target/')
            yield CustomDataset(self.source), CustomDataset(self.target)
            os.remove(os.path.join(self.target, target))
        else:
            shutil.rmtree('source/')
            shutil.rmtree('target/')


if __name__ == '__main__':
    x = DatasetGenerator("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\testing_consistory\\1")
    for a, b in x:
        print(str(a))
        print(str(b))
