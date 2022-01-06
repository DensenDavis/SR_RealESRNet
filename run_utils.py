def parameter_count(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_non_trainable_params = pytorch_total_params - pytorch_total_trainable_params
    print(f'Total parameters = {pytorch_total_params}\n Tainable parameters = {pytorch_total_trainable_params}\n Non Trainable parametrs = {pytorch_non_trainable_params}\n')
    return None