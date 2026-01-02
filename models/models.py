
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'posenet' or opt.model == 'poselstm' or opt.model == 'posetransformer' or opt.model == 'poseseparate':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (opt.model))
    return model
