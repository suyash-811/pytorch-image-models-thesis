import timm
import torch


torch.cuda.empty_cache()
model = timm.create_model("deit_base_distilled_patch16_224",
                          in_chans=1,
                          num_classes=5,
                          multiclass=True)

x = torch.rand((2,1,224,224))

out, attn = model(x)
print(out)