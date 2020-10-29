### Imagenet Evaluation

Place your `train` and `val` directories in `data`. 

Example commands: 
```bash
# Evaluate ViT on CPU
python main.py data -e -a 'B_16_imagenet1k' --vit --pretrained

# Evaluate ViT on GPU
python main.py data -e -a 'L_32_imagenet1k' --vit --pretrained --gpu 0 
```