"""
wget http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list
"""

from pathlib import Path
from nltk.corpus import wordnet

classes_file = 'imagenet.synset.obtain_synset_list'
output_file = 'labels_map_21k.txt'
with open(Path(classes_file)) as f:
    classes = f.read().splitlines()
    classes = [c for c in classes if c != '']
    assert len(classes) in [1000, 21_841]
    classes = [wordnet.synset_from_pos_and_offset('n', int(c[1:])).lemmas()[0].name() for c in classes]
with open(output_file, 'w') as f:
    print('\n'.join(classes), file=f)
    