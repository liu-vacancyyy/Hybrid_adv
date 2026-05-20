import os

cur_dir=os.path.dirname(os.path.abspath(__file__))
last_dir=os.path.dirname(cur_dir)

txts_dir=os.path.join(last_dir,'txts')

if not os.path.exists(txts_dir):
    os.makedirs(txts_dir)

file_path=os.path.join(txts_dir,'1.txt')
print(file_path)
# with open(file_path, mode='w', encoding='utf-8') as f:
#     f.write("6!")