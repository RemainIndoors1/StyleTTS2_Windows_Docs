# round 2 Training
def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print(f'{key} loaded')
            try:
                model[key].load_state_dict(params[key], strict=True)
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for (k_m, _), (k_c, v_c) in zip(model[key].state_dict().items(), state_dict.items()):
                    new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
    _ = [model[key].eval() for key in model]
    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
    return model, optimizer, epoch, iters