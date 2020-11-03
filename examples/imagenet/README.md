### Imagenet Evaluation

Place your `train` and `val` directories in `data`. 

Example commands: 
```bash
# Evaluate ViT on CPU
python main.py data -e -a 'B_16_imagenet1k' --vit --pretrained -b 16 --image_size 384

# Evaluate ViT on GPU
python main.py data -e -a 'B_16_imagenet1k' --vit --pretrained -b 16 --image_size 384 --gpu 0 
```