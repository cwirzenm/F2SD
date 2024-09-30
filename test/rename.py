import os

root_dir = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\temporalstory\\flintstones\\gt_flintstones"
dirs = os.listdir(root_dir)

count = 1
for d in dirs:
    curr_dir = os.path.join(root_dir, d)
    for f in os.listdir(curr_dir):
        split = f.split('-')
        new_name = f"{split[0]}-{d}-{split[2]}-{split[3]}"
        count += 1
        print(f"Rename {os.path.join(curr_dir, f)} to {os.path.join(root_dir, new_name)}")
        os.rename(os.path.join(curr_dir, f), os.path.join(root_dir, new_name))
