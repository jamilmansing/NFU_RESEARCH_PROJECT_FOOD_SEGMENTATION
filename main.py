import torch
import transformers


print("PyTorch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)
print("Is GPU available?:", torch.cuda.is_available())